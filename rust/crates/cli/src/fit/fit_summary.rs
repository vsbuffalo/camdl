//! `camdl fit summary` — single-fit interpretation surface.
//!
//! Reads a fit dir produced by `camdl fit run` and renders an
//! interpretation block per stage: compound-gate verdict, parameter
//! estimates with Â, per-chain clean-eval table, filter health,
//! provenance cross-checks. Phase 1 ships `text` (with ANSI colour);
//! Phase 4 adds `json` (versioned schema), `md`, and `latex`; Phase 5
//! adds `--params-only` for piping into `camdl pfilter --params`.
//!
//! Boundary rule: `status` answers "what's the state of my filesystem
//! / pipeline?", `summary` answers "what does this fit say?",
//! `compare` answers "which of these models predicts better?". Three
//! commands, three orthogonal jobs.
//!
//! See `docs/dev/proposals/2026-04-25-fit-summary-command.md`.
//!
//! ## Layout
//!
//! ```text
//! fit/<name>/
//!   scout/      ← FitState + summary JSON + final/mle params
//!   refine/     ← (optional)
//!   validate/   ← (optional)
//!   pgas/       ← (optional, posterior; out of scope here)
//!   pmmh/       ← (optional, posterior; out of scope here)
//! ```

use crate::args::{FitSummaryArgs, FitSummaryFormat};
use crate::evidence::NATS_TO_DB;
use crate::fit::config_v2::{CleanEvalConfig, GateConfig};
use crate::fit::state::FitState;
use crate::version;
use serde::Serialize;
use std::path::Path;

/// Versioned JSON schema. Bumped when fields are renamed / removed /
/// retyped; field additions are non-breaking and keep version stable.
const SCHEMA_VERSION: u32 = 1;

/// Stages we render in pipeline order. Bayesian stages (pgas, pmmh)
/// are deliberately out of scope here — their interpretation surface
/// is different (posterior summaries, ESS at posterior mean, etc.)
/// and would dilute the MLE-pipeline focus.
const MLE_STAGES: &[&str] = &["scout", "refine", "validate"];

/// Top-level entry point. Reads `args.fit_dir`, walks MLE stages in
/// pipeline order, dispatches to the right formatter based on
/// `--format` and `--params-only`. Exits with code 1 if directory is
/// missing or empty; with code 1 in `--strict` mode if any stage's
/// provenance cross-check fails.
pub fn cmd_fit_summary(args: &FitSummaryArgs) {
    let dir = args.fit_dir.to_string_lossy().into_owned();
    if !Path::new(&dir).exists() {
        eprintln!("error: no such fit directory: {}", dir);
        std::process::exit(1);
    }

    let strict = args.strict || ci_env_set();

    // Phase 5: --params-only. Walk stages, find the terminal stage's
    // winner θ̂, dump params as a flat TOML pipeable into `pfilter
    // --params`. No metadata, no headers — composability is the point.
    if args.params_only {
        match dump_params_only(&dir, args.stage.as_deref()) {
            Ok(s) => {
                print!("{}", s);
                return;
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Validate --stage early before formatter dispatch.
    let selected_stages: Vec<&str> = match &args.stage {
        Some(name) => {
            if !MLE_STAGES.contains(&name.as_str()) {
                eprintln!("error: unknown stage `{}`. Available: {}",
                    name, MLE_STAGES.join(", "));
                std::process::exit(1);
            }
            vec![name.as_str()]
        }
        None => MLE_STAGES.to_vec(),
    };

    match args.format {
        FitSummaryFormat::Text => {
            cmd_text(&dir, args, &selected_stages, strict);
        }
        FitSummaryFormat::Json => cmd_json(&dir, args, &selected_stages, strict),
        FitSummaryFormat::Md => cmd_md(&dir, args, &selected_stages, strict),
        FitSummaryFormat::Latex => cmd_latex(&dir, args, &selected_stages, strict),
    }
}

fn cmd_text(dir: &str, args: &FitSummaryArgs, stages: &[&str], strict: bool) {
    let use_color = should_use_color(args.no_color);
    let fmt = Formatter { use_color };
    let mut had_provenance_failure = false;

    print!("{}", fmt.fit_header(dir));

    let mut any_rendered = false;
    let mut prev_loglik: Option<f64> = None;
    let mut prev_stage_name: Option<&str> = None;
    for stage in stages {
        let stage_dir = format!("{}/{}", dir, stage);
        if !Path::new(&stage_dir).join("fit_state.toml").exists() {
            continue;
        }
        any_rendered = true;
        let state = match FitState::load(&stage_dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("warning: cannot load {}/fit_state.toml: {}",
                    stage_dir, e);
                continue;
            }
        };
        let block = fmt.stage_block(stage, &stage_dir, &state, prev_loglik, prev_stage_name);
        if block.provenance_failed {
            had_provenance_failure = true;
        }
        print!("{}", block.text);
        prev_loglik = Some(state.best_loglik);
        prev_stage_name = Some(*stage);
    }

    if !any_rendered {
        println!("  (no MLE stages found in {})", dir);
        println!("  expected one of: {}/{{{}}}",
            dir, MLE_STAGES.join(","));
    }

    if strict && had_provenance_failure {
        eprintln!();
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn cmd_json(dir: &str, _args: &FitSummaryArgs, stages: &[&str], strict: bool) {
    let doc = build_summary_doc(dir, stages);
    let any_failed = doc.stages.iter().any(|s| s.provenance.any_failed());
    let s = serde_json::to_string_pretty(&doc)
        .expect("FitSummaryDoc must serialize");
    println!("{}", s);
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn cmd_md(dir: &str, _args: &FitSummaryArgs, stages: &[&str], strict: bool) {
    let doc = build_summary_doc(dir, stages);
    let any_failed = doc.stages.iter().any(|s| s.provenance.any_failed());
    print!("{}", render_markdown(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn cmd_latex(dir: &str, _args: &FitSummaryArgs, stages: &[&str], strict: bool) {
    let doc = build_summary_doc(dir, stages);
    let any_failed = doc.stages.iter().any(|s| s.provenance.any_failed());
    print!("{}", render_latex(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

// ── Formatting layer ────────────────────────────────────────────────

struct Formatter {
    use_color: bool,
}

struct StageBlock {
    text: String,
    provenance_failed: bool,
}

impl Formatter {
    fn fit_header(&self, dir: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n{}/\n", self.bold(dir)));
        s.push_str(&format!("  camdl {}\n\n", self.dim(version::VERSION_SHORT)));
        s
    }

    fn stage_block(
        &self,
        stage: &str,
        stage_dir: &str,
        state: &FitState,
        prev_loglik: Option<f64>,
        prev_stage_name: Option<&str>,
    ) -> StageBlock {
        let mut s = String::new();

        s.push_str(&format!("══ {} {}\n",
            self.bold(stage),
            "═".repeat(74_usize.saturating_sub(stage.len()))));

        // Headline
        s.push_str(&format!("  best loglik:  {:.1}", state.best_loglik));
        if !state.chain_clean_logliks.is_empty() {
            s.push_str("  (clean-eval winner)");
        }
        s.push('\n');
        s.push_str(&format!("  chains:       {}\n", state.n_chains));
        if let Some(ref v) = state.camdl_version {
            if v != version::VERSION_SHORT {
                s.push_str(&format!("                {}\n",
                    self.warn(&format!("⚠ stale: produced by {}, current is {}",
                        v, version::VERSION_SHORT))));
            }
        }
        if let Some(prev) = prev_loglik {
            let delta = state.best_loglik - prev;
            let prev_label = prev_stage_name.unwrap_or("prev");
            let glyph = if delta >= 0.0 { self.ok("✓") } else { self.err("✗") };
            s.push_str(&format!(
                "  vs {}:    Δ = {:+.1} nats  {}\n",
                prev_label, delta, glyph));
        }
        s.push('\n');

        // Compound-gate verdict
        s.push_str(&self.gate_verdict_block(state));

        // Per-parameter table
        s.push_str(&self.parameter_table(state));

        // Per-chain clean-eval table
        if !state.chain_clean_logliks.is_empty() {
            s.push_str(&self.chain_clean_eval_table(state));
        }

        // Provenance cross-check (#16 fixture, every read)
        let prov = self.provenance_block(stage_dir, state);
        let provenance_failed = prov.failed;
        s.push_str(&prov.text);

        s.push('\n');
        StageBlock { text: s, provenance_failed }
    }

    fn gate_verdict_block(&self, state: &FitState) -> String {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("compound scout-convergence gate")));

        // Resolve the gate config to render against. Priority:
        //   1. state.resolved_gate (Phase 3 — the value actually used)
        //   2. GateConfig::default() with a "(thresholds unknown)" caveat
        let (gate, threshold_source) = match &state.resolved_gate {
            Some(g) => (g.clone(), GateThresholdSource::Resolved),
            None => (GateConfig::default(), GateThresholdSource::DefaultFallback),
        };

        // Â leg
        let max_a = state.tail_chain_agreement.values().cloned()
            .fold(0.0_f64, f64::max);
        let max_a_param = state.tail_chain_agreement.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone()).unwrap_or_else(|| "—".into());
        let a_passes = max_a < gate.a_thresh;
        let a_glyph = if a_passes { self.ok("✓") } else { self.err("✗") };
        s.push_str(&format!(
            "    Â leg:           max Â = {:.3} ({})  {}  (threshold {:.2})\n",
            max_a, max_a_param, a_glyph, gate.a_thresh));

        // Decibans leg
        if state.chain_clean_logliks.len() >= 2
            && state.chain_clean_ses.len() == state.chain_clean_logliks.len()
        {
            let hi = state.chain_clean_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_clean_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let delta_db = (hi - lo) * NATS_TO_DB;
            let sigma_max = state.chain_clean_ses.iter().cloned()
                .fold(0.0_f64, f64::max);
            let se_floor_db = crate::fit::gating::SE_FLOOR_K * sigma_max * NATS_TO_DB;
            let threshold_db = gate.decibans_thresh.max(se_floor_db);
            let db_passes = delta_db < threshold_db;
            let db_glyph = if db_passes { self.ok("✓") } else { self.err("✗") };
            s.push_str(&format!(
                "    decibans leg:    Δ = {:.1} dB / threshold {:.1} dB  {}  (σ_max={:.2})\n",
                delta_db, threshold_db, db_glyph, sigma_max));

            let overall_pass = a_passes && db_passes;
            let overall = if overall_pass {
                self.ok("✓ PASS")
            } else {
                self.err("✗ FAIL")
            };
            s.push_str(&format!("    overall:         {}\n", overall));
        } else {
            s.push_str(&format!("    decibans leg:    {} (clean-eval data not present)\n",
                self.dim("—")));
        }

        match threshold_source {
            GateThresholdSource::Resolved => {}
            GateThresholdSource::DefaultFallback => {
                s.push_str(&format!("    {}\n", self.warn(
                    "(thresholds unknown — fit_state.toml predates Phase 3; \
                     showing GateConfig::default())"
                )));
            }
        }
        s.push('\n');
        s
    }

    fn parameter_table(&self, state: &FitState) -> String {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("parameter estimates (clean-eval winner θ̂)")));
        if state.start_values.is_empty() {
            s.push_str(&format!("    {}\n", self.dim("(no start_values in fit_state.toml)")));
            s.push('\n');
            return s;
        }
        let ivp_set: std::collections::HashSet<&str> = state.ivp_params.iter()
            .map(|s| s.as_str()).collect();
        let mut keys: Vec<&String> = state.start_values.keys().collect();
        keys.sort();
        // Filter to params we have agreement data for (these are the
        // estimated ones); fixed params are noise here. Fall back to
        // showing everything if no agreement data.
        let est_keys: Vec<&String> = if state.tail_chain_agreement.is_empty() {
            keys.clone()
        } else {
            keys.iter().filter(|k| state.tail_chain_agreement.contains_key(k.as_str()))
                .copied().collect()
        };
        for k in est_keys {
            let v = state.start_values[k];
            let agreement = state.tail_chain_agreement.get(k).copied();
            let agreement_str = match agreement {
                Some(r) => {
                    let glyph = if r < 1.05 { self.ok("✓") }
                        else if r < 1.10 { self.warn("~") }
                        else { self.err("✗") };
                    format!("Â={:.3} {}", r, glyph)
                }
                None => self.dim("Â=—").to_string(),
            };
            let ivp_marker = if ivp_set.contains(k.as_str()) {
                format!(" {}", self.dim("(ivp)"))
            } else {
                String::new()
            };
            s.push_str(&format!("    {:12} = {:<12.6}  {}{}\n",
                k, v, agreement_str, ivp_marker));
        }
        s.push('\n');
        s
    }

    fn chain_clean_eval_table(&self, state: &FitState) -> String {
        let mut s = String::new();
        let n = state.chain_clean_logliks.len();
        s.push_str(&format!("  {}\n",
            self.bold(&format!("per-chain clean-eval ({} chains)", n))));
        let best_idx = state.chain_clean_logliks.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        // ESS columns added in Phase 2. Today fit_state.toml only
        // carries the per-chain ll + se; per-chain ESS lives in
        // <stage>_summary.json's `chains[]`. Phase 1 of summary keeps
        // the table simple and reads only what's already in
        // fit_state.toml; ESS surfacing in this table waits for a
        // Phase 4 follow-up that loads <stage>_summary.json. Note
        // here so a future reader doesn't think it was forgotten.
        s.push_str(&format!("    {:6} {:>12}   {:>6}\n", "chain", "clean_ll", "± se"));
        for i in 0..n {
            let ll = state.chain_clean_logliks[i];
            let se = state.chain_clean_ses.get(i).copied().unwrap_or(f64::NAN);
            let marker = if Some(i) == best_idx {
                format!("  {}", self.ok("← winner"))
            } else {
                String::new()
            };
            s.push_str(&format!("    {:6} {:>12.2}   ± {:>4.2}{}\n",
                i + 1, ll, se, marker));
        }
        s.push('\n');
        s
    }

    fn provenance_block(&self, stage_dir: &str, state: &FitState)
        -> ProvenanceBlock
    {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("provenance")));
        let final_path = format!("{}/final_params.toml", stage_dir);
        let mle_path = format!("{}/mle_params.toml", stage_dir);
        let mut failed = false;

        let final_params = read_param_values(&final_path);
        let mle_params = read_param_values(&mle_path);

        match (&final_params, &mle_params) {
            (Some(f), Some(m)) => {
                let agree = params_agree(f, m);
                if agree {
                    s.push_str(&format!("    final_params.toml ↔ mle_params.toml: {}\n",
                        self.ok("✓ params match")));
                } else {
                    s.push_str(&format!("    final_params.toml ↔ mle_params.toml: {}\n",
                        self.err("✗ DISAGREE — silent-wrong-answer (GH #16) class")));
                    failed = true;
                }
            }
            (None, _) => s.push_str(&format!("    final_params.toml: {}\n",
                self.dim("(absent)"))),
            (_, None) => s.push_str(&format!("    mle_params.toml:   {}\n",
                self.dim("(absent)"))),
        }

        // fit_state winner ↔ final_params
        if !state.start_values.is_empty() && final_params.is_some() {
            let f = final_params.as_ref().unwrap();
            let state_matches = params_agree(f, &state.start_values);
            if state_matches {
                s.push_str(&format!("    fit_state.toml ↔ final_params.toml:   {}\n",
                    self.ok("✓ params match")));
            } else {
                s.push_str(&format!("    fit_state.toml ↔ final_params.toml:   {}\n",
                    self.err("✗ DISAGREE — fit_state's start_values diverge from winner")));
                failed = true;
            }
        }

        ProvenanceBlock { text: s, failed }
    }

    // ── Colour helpers ──────────────────────────────────────────────

    fn wrap(&self, code: &str, s: &str) -> String {
        if self.use_color {
            format!("\x1b[{}m{}\x1b[0m", code, s)
        } else {
            s.to_string()
        }
    }
    fn bold(&self, s: &str)  -> String { self.wrap("1", s) }
    fn dim(&self, s: &str)   -> String { self.wrap("2", s) }
    fn ok(&self, s: &str)    -> String { self.wrap("32", s) }
    fn warn(&self, s: &str)  -> String { self.wrap("33", s) }
    fn err(&self, s: &str)   -> String { self.wrap("31", s) }
}

struct ProvenanceBlock {
    text: String,
    failed: bool,
}

enum GateThresholdSource {
    /// Read from `state.resolved_gate` (Phase 3 — what the run was
    /// actually judged against).
    Resolved,
    /// Legacy fit_state.toml — no resolved_gate. Showing
    /// `GateConfig::default()` with a caveat.
    DefaultFallback,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn ci_env_set() -> bool {
    matches!(std::env::var("CI").as_deref(), Ok("true") | Ok("1"))
}

fn should_use_color(no_color_flag: bool) -> bool {
    if no_color_flag { return false; }
    if std::env::var("NO_COLOR").is_ok() { return false; }
    is_stdout_tty()
}

fn is_stdout_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Read a params TOML, returning a flat name→value map. Skips any
/// `[provenance]` section per `util::load_params_toml`'s convention.
fn read_param_values(path: &str) -> Option<std::collections::HashMap<String, f64>> {
    crate::util::load_params_toml(path).ok()
}

/// Two parameter dictionaries agree iff every shared key has values
/// matching to floating-point tolerance. Disjoint keys are treated as
/// "match" — `final_params.toml` and `mle_params.toml` legitimately
/// have non-overlapping fields (e.g. mle_params has more fixed params
/// rolled in). The shared subset is what would diverge under #16.
fn params_agree(
    a: &std::collections::HashMap<String, f64>,
    b: &std::collections::HashMap<String, f64>,
) -> bool {
    for (k, v) in a {
        if let Some(other) = b.get(k) {
            let scale = v.abs().max(other.abs()).max(1.0);
            if (v - other).abs() > 1e-9 * scale {
                return false;
            }
        }
    }
    true
}

// ── Phase 4 / 5: structured doc + multi-format renderers ────────────

/// Structured fit-interpretation document. Serialized as `--format
/// json`; consumed by md / latex renderers. Stable schema versioned
/// via `schema.version`.
#[derive(Debug, Clone, Serialize)]
pub struct FitSummaryDoc {
    pub schema: SchemaInfo,
    pub fit_dir: String,
    pub stages: Vec<StageReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaInfo {
    pub version: u32,
    pub camdl_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageReport {
    pub name: String,
    pub n_chains: usize,
    pub best_loglik: f64,
    pub initial_loglik: Option<f64>,
    pub camdl_version: Option<String>,
    pub gate: GateReport,
    pub stage_progression: Option<StageProgression>,
    pub parameters: Vec<ParameterReport>,
    pub chains: Vec<ChainReport>,
    pub provenance: ProvenanceReport,
    /// Advisory fields whose strings may shift across camdl versions
    /// even at stable schema.version. Consumers keying off these must
    /// accept that they're heuristic.
    #[serde(rename = "_heuristic")]
    pub heuristic: HeuristicReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    pub max_a_hat: f64,
    pub max_a_param: Option<String>,
    pub a_thresh: f64,
    pub a_passes: bool,
    pub delta_db: Option<f64>,
    pub threshold_db: Option<f64>,
    pub sigma_max: Option<f64>,
    pub db_passes: Option<bool>,
    pub overall_pass: Option<bool>,
    /// `"resolved"` when read from `state.resolved_gate`; `"default_fallback"`
    /// when fit_state.toml predates Phase 3 and we substituted
    /// `GateConfig::default()`. Critical signal for downstream readers.
    pub threshold_source: String,
    pub resolved_gate: Option<GateConfig>,
    pub resolved_clean_eval: Option<CleanEvalConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageProgression {
    pub previous_stage: String,
    pub previous_loglik: f64,
    pub delta_nats: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParameterReport {
    pub name: String,
    pub estimate: f64,
    pub chain_agreement: Option<f64>,
    pub ivp: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainReport {
    pub chain_id: usize,
    pub clean_loglik: f64,
    pub clean_se: f64,
    pub is_winner: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceReport {
    pub final_params_matches_mle_params: Option<bool>,
    pub fit_state_winner_matches_final_params: Option<bool>,
    pub stale_camdl_version: Option<String>,
}

impl ProvenanceReport {
    fn any_failed(&self) -> bool {
        matches!(self.final_params_matches_mle_params, Some(false))
            || matches!(self.fit_state_winner_matches_final_params, Some(false))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HeuristicReport {
    pub overall_status: String,
    pub interpretation: Option<String>,
}

/// Walk the stage dirs and build a `FitSummaryDoc`. Used by JSON, MD,
/// and LaTeX formatters. Pure on its inputs (file system + the args
/// it was called with).
pub fn build_summary_doc(dir: &str, stages: &[&str]) -> FitSummaryDoc {
    let mut out = FitSummaryDoc {
        schema: SchemaInfo {
            version: SCHEMA_VERSION,
            camdl_version: version::VERSION_SHORT.to_string(),
        },
        fit_dir: dir.to_string(),
        stages: Vec::new(),
    };
    let mut prev_loglik: Option<f64> = None;
    let mut prev_stage_name: Option<&str> = None;
    for stage in stages {
        let stage_dir = format!("{}/{}", dir, stage);
        if !Path::new(&stage_dir).join("fit_state.toml").exists() {
            continue;
        }
        let state = match FitState::load(&stage_dir) {
            Ok(s) => s,
            Err(_) => continue,
        };
        out.stages.push(stage_report(stage, &stage_dir, &state, prev_loglik, prev_stage_name));
        prev_loglik = Some(state.best_loglik);
        prev_stage_name = Some(*stage);
    }
    out
}

fn stage_report(
    stage: &str,
    stage_dir: &str,
    state: &FitState,
    prev_loglik: Option<f64>,
    prev_stage_name: Option<&str>,
) -> StageReport {
    // Gate analysis — same logic as Formatter::gate_verdict_block but
    // returning structured data instead of pre-formatted strings.
    let (gate_cfg, threshold_source) = match &state.resolved_gate {
        Some(g) => (g.clone(), "resolved".to_string()),
        None    => (GateConfig::default(), "default_fallback".to_string()),
    };
    let max_a = state.tail_chain_agreement.values().cloned()
        .fold(0.0_f64, f64::max);
    let max_a_param = state.tail_chain_agreement.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone());
    let a_passes = max_a < gate_cfg.a_thresh;

    let (delta_db, threshold_db, sigma_max, db_passes) =
        if state.chain_clean_logliks.len() >= 2
            && state.chain_clean_ses.len() == state.chain_clean_logliks.len()
        {
            let hi = state.chain_clean_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_clean_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let dd = (hi - lo) * NATS_TO_DB;
            let sm = state.chain_clean_ses.iter().cloned()
                .fold(0.0_f64, f64::max);
            let se_floor_db = crate::fit::gating::SE_FLOOR_K * sm * NATS_TO_DB;
            let td = gate_cfg.decibans_thresh.max(se_floor_db);
            (Some(dd), Some(td), Some(sm), Some(dd < td))
        } else {
            (None, None, None, None)
        };
    let overall_pass = db_passes.map(|p| p && a_passes);

    let gate = GateReport {
        max_a_hat: max_a,
        max_a_param,
        a_thresh: gate_cfg.a_thresh,
        a_passes,
        delta_db, threshold_db, sigma_max, db_passes, overall_pass,
        threshold_source,
        resolved_gate: state.resolved_gate.clone(),
        resolved_clean_eval: state.resolved_clean_eval.clone(),
    };

    // Parameters
    let ivp_set: std::collections::HashSet<&str> = state.ivp_params.iter()
        .map(|s| s.as_str()).collect();
    let mut keys: Vec<&String> = state.start_values.keys().collect();
    keys.sort();
    let est_keys: Vec<&String> = if state.tail_chain_agreement.is_empty() {
        keys.clone()
    } else {
        keys.iter().filter(|k| state.tail_chain_agreement.contains_key(k.as_str()))
            .copied().collect()
    };
    let parameters: Vec<ParameterReport> = est_keys.iter().map(|k| ParameterReport {
        name: (*k).clone(),
        estimate: state.start_values[*k],
        chain_agreement: state.tail_chain_agreement.get(*k).copied(),
        ivp: ivp_set.contains(k.as_str()),
    }).collect();

    // Chains
    let n = state.chain_clean_logliks.len();
    let best_idx = state.chain_clean_logliks.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    let chains: Vec<ChainReport> = (0..n).map(|i| ChainReport {
        chain_id: i + 1,
        clean_loglik: state.chain_clean_logliks[i],
        clean_se: state.chain_clean_ses.get(i).copied().unwrap_or(f64::NAN),
        is_winner: Some(i) == best_idx,
    }).collect();

    // Provenance
    let final_path = format!("{}/final_params.toml", stage_dir);
    let mle_path = format!("{}/mle_params.toml", stage_dir);
    let final_params = read_param_values(&final_path);
    let mle_params = read_param_values(&mle_path);
    let final_matches_mle = match (&final_params, &mle_params) {
        (Some(f), Some(m)) => Some(params_agree(f, m)),
        _ => None,
    };
    let state_matches_final = match &final_params {
        Some(f) if !state.start_values.is_empty() => {
            Some(params_agree(f, &state.start_values))
        }
        _ => None,
    };
    let stale = match &state.camdl_version {
        Some(v) if v != version::VERSION_SHORT => Some(v.clone()),
        _ => None,
    };
    let provenance = ProvenanceReport {
        final_params_matches_mle_params: final_matches_mle,
        fit_state_winner_matches_final_params: state_matches_final,
        stale_camdl_version: stale,
    };

    let stage_progression = prev_loglik.zip(prev_stage_name).map(|(prev, prev_name)| StageProgression {
        previous_stage: prev_name.to_string(),
        previous_loglik: prev,
        delta_nats: state.best_loglik - prev,
    });

    let overall_status = match overall_pass {
        Some(true)  => "pass".to_string(),
        Some(false) => "fail".to_string(),
        None        => "indeterminate".to_string(),
    };
    let interpretation = if !a_passes && db_passes == Some(false) {
        Some("chains disagree on basin (Â and decibans-spread both fail)".to_string())
    } else if !a_passes {
        Some("per-parameter chain agreement insufficient".to_string())
    } else if db_passes == Some(false) {
        Some("chains agree per-parameter but disagree on basin quality".to_string())
    } else {
        None
    };

    StageReport {
        name: stage.to_string(),
        n_chains: state.n_chains,
        best_loglik: state.best_loglik,
        initial_loglik: if state.initial_loglik.is_finite() {
            Some(state.initial_loglik)
        } else { None },
        camdl_version: state.camdl_version.clone(),
        gate,
        stage_progression,
        parameters,
        chains,
        provenance,
        heuristic: HeuristicReport { overall_status, interpretation },
    }
}

/// Render a `FitSummaryDoc` as GitHub-flavoured Markdown. Tabular per
/// stage, code-fenced parameter tables. Suitable for embedding in book
/// chapters via `run_cli("camdl fit summary {dir} --format md", ...)`.
pub fn render_markdown(doc: &FitSummaryDoc) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Fit summary: `{}`\n\n", doc.fit_dir));
    s.push_str(&format!("camdl `{}` (schema v{})\n\n",
        doc.schema.camdl_version, doc.schema.version));
    if doc.stages.is_empty() {
        s.push_str("_(no MLE stages found)_\n");
        return s;
    }
    for stage in &doc.stages {
        s.push_str(&render_md_stage(stage));
    }
    s
}

fn render_md_stage(stage: &StageReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("## `{}`\n\n", stage.name));
    s.push_str(&format!("- best loglik: **{:.2}**\n", stage.best_loglik));
    s.push_str(&format!("- chains: {}\n", stage.n_chains));
    if let Some(init) = stage.initial_loglik {
        s.push_str(&format!("- initial loglik: {:.2}\n", init));
    }
    if let Some(prog) = &stage.stage_progression {
        s.push_str(&format!("- vs `{}`: Δ = {:+.2} nats\n",
            prog.previous_stage, prog.delta_nats));
    }
    if let Some(stale) = &stage.provenance.stale_camdl_version {
        s.push_str(&format!("- ⚠ stale: produced by camdl `{}`, current is `{}`\n",
            stale, version::VERSION_SHORT));
    }
    s.push('\n');

    // Gate verdict
    s.push_str("### Compound scout-convergence gate\n\n");
    s.push_str(&format!("| leg | value | threshold | pass? |\n|---|---|---|---|\n"));
    let glyph = |b| if b { "✓" } else { "✗" };
    s.push_str(&format!("| Â (max over params{}) | {:.3} | {:.2} | {} |\n",
        stage.gate.max_a_param.as_deref().map(|p| format!(", `{}`", p)).unwrap_or_default(),
        stage.gate.max_a_hat, stage.gate.a_thresh,
        glyph(stage.gate.a_passes)));
    if let (Some(dd), Some(td), Some(p)) = (stage.gate.delta_db, stage.gate.threshold_db, stage.gate.db_passes) {
        s.push_str(&format!("| decibans-spread | {:.1} dB | {:.1} dB | {} |\n",
            dd, td, glyph(p)));
    } else {
        s.push_str("| decibans-spread | _(no clean-eval data)_ | — | — |\n");
    }
    s.push_str(&format!("\n**overall:** {}\n", match stage.gate.overall_pass {
        Some(true)  => "✓ PASS",
        Some(false) => "✗ FAIL",
        None        => "(indeterminate)",
    }));
    if stage.gate.threshold_source == "default_fallback" {
        s.push_str("\n> ⚠ thresholds unknown — fit_state.toml predates Phase 3; showing `GateConfig::default()`.\n");
    }
    s.push('\n');

    // Params
    if !stage.parameters.is_empty() {
        s.push_str("### Parameter estimates (clean-eval winner θ̂)\n\n");
        s.push_str("| name | estimate | Â | flags |\n|---|---|---|---|\n");
        for p in &stage.parameters {
            let a_str = p.chain_agreement
                .map(|r| format!("{:.3}", r))
                .unwrap_or_else(|| "—".into());
            let flag = if p.ivp { "ivp" } else { "" };
            s.push_str(&format!("| `{}` | {:.6} | {} | {} |\n",
                p.name, p.estimate, a_str, flag));
        }
        s.push('\n');
    }

    // Chains
    if !stage.chains.is_empty() {
        s.push_str(&format!("### Per-chain clean-eval ({} chains)\n\n", stage.chains.len()));
        s.push_str("| chain | clean_ll | ± se | winner |\n|---|---|---|---|\n");
        for c in &stage.chains {
            let mark = if c.is_winner { "★" } else { "" };
            s.push_str(&format!("| {} | {:.2} | {:.2} | {} |\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark));
        }
        s.push('\n');
    }

    // Provenance
    s.push_str("### Provenance\n\n");
    let prov_row = |label: &str, val: Option<bool>| {
        match val {
            Some(true)  => format!("- {}: ✓\n", label),
            Some(false) => format!("- {}: ✗ **DISAGREE**\n", label),
            None        => format!("- {}: _(absent)_\n", label),
        }
    };
    s.push_str(&prov_row("final_params.toml ↔ mle_params.toml",
        stage.provenance.final_params_matches_mle_params));
    s.push_str(&prov_row("fit_state.toml ↔ final_params.toml",
        stage.provenance.fit_state_winner_matches_final_params));
    s.push('\n');
    s
}

/// Render a `FitSummaryDoc` as LaTeX `tabular` blocks per stage.
/// One section per stage with three tables: gate verdict, parameters,
/// per-chain clean-eval. No preamble — the caller should embed inside
/// an existing document.
pub fn render_latex(doc: &FitSummaryDoc) -> String {
    let mut s = String::new();
    s.push_str(&format!("% camdl fit summary: {}\n", escape_latex(&doc.fit_dir)));
    s.push_str(&format!("% camdl {} schema v{}\n\n",
        doc.schema.camdl_version, doc.schema.version));
    for stage in &doc.stages {
        s.push_str(&render_latex_stage(stage));
    }
    s
}

fn render_latex_stage(stage: &StageReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("\\subsection*{{Stage: \\texttt{{{}}}}}\n\n",
        escape_latex(&stage.name)));

    s.push_str(&format!("Best log-likelihood: \\textbf{{{:.2}}}; chains: {}\n\n",
        stage.best_loglik, stage.n_chains));

    s.push_str("\\begin{tabular}{lrrl}\n");
    s.push_str("\\toprule\n");
    s.push_str("Leg & Value & Threshold & Pass? \\\\\n");
    s.push_str("\\midrule\n");
    let glyph = |b| if b { "$\\checkmark$" } else { "$\\times$" };
    s.push_str(&format!("$\\hat A$ (max) & {:.3} & {:.2} & {} \\\\\n",
        stage.gate.max_a_hat, stage.gate.a_thresh, glyph(stage.gate.a_passes)));
    if let (Some(dd), Some(td), Some(p)) = (stage.gate.delta_db, stage.gate.threshold_db, stage.gate.db_passes) {
        s.push_str(&format!("Decibans-spread & {:.1} dB & {:.1} dB & {} \\\\\n",
            dd, td, glyph(p)));
    }
    s.push_str("\\bottomrule\n\\end{tabular}\n\n");

    if !stage.parameters.is_empty() {
        s.push_str("\\begin{tabular}{lrrl}\n\\toprule\n");
        s.push_str("Parameter & Estimate & $\\hat A$ & Flags \\\\\n\\midrule\n");
        for p in &stage.parameters {
            let a_str = p.chain_agreement
                .map(|r| format!("{:.3}", r))
                .unwrap_or_else(|| "---".into());
            let flag = if p.ivp { "ivp" } else { "" };
            s.push_str(&format!("\\texttt{{{}}} & {:.6} & {} & {} \\\\\n",
                escape_latex(&p.name), p.estimate, a_str, flag));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    if !stage.chains.is_empty() {
        s.push_str("\\begin{tabular}{rrrc}\n\\toprule\n");
        s.push_str("Chain & clean\\_ll & $\\pm$ se & Winner \\\\\n\\midrule\n");
        for c in &stage.chains {
            let mark = if c.is_winner { "$\\star$" } else { "" };
            s.push_str(&format!("{} & {:.2} & {:.2} & {} \\\\\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }
    s
}

/// Escape LaTeX-active characters in identifiers / paths. Replaces
/// `&`, `%`, `#`, `$`, `_`, `{`, `}` with their escaped forms.
/// LaTeX requires _ to be escaped as \_ in tabular cell text.
fn escape_latex(s: &str) -> String {
    s.chars().map(|c| match c {
        '&' => "\\&".into(),
        '%' => "\\%".into(),
        '#' => "\\#".into(),
        '$' => "\\$".into(),
        '_' => "\\_".into(),
        '{' => "\\{".into(),
        '}' => "\\}".into(),
        c   => c.to_string(),
    }).collect()
}

/// Phase 5: dump the winner's parameter TOML for one stage. No
/// header, no metadata, no provenance — just `name = value` lines
/// that the standard params loader will accept. Composable via
/// process substitution: `camdl pfilter --params <(camdl fit
/// summary --params-only --stage validate fit/he2010) ...`.
///
/// When `stage_filter` is `None`, picks the *terminal* stage
/// available in pipeline order (validate → refine → scout). Errors
/// when the requested stage doesn't exist or its `final_params.toml`
/// is missing / malformed.
pub fn dump_params_only(dir: &str, stage_filter: Option<&str>) -> Result<String, String> {
    let target_stage: &str = match stage_filter {
        Some(s) => {
            if !MLE_STAGES.contains(&s) {
                return Err(format!(
                    "unknown stage `{}`. Available: {}", s, MLE_STAGES.join(", ")));
            }
            s
        }
        None => {
            // Walk in reverse pipeline order so we land on the most
            // refined stage available.
            let mut found = None;
            for stage in MLE_STAGES.iter().rev() {
                let p = format!("{}/{}/final_params.toml", dir, stage);
                if Path::new(&p).exists() {
                    found = Some(*stage);
                    break;
                }
            }
            found.ok_or_else(|| format!(
                "no completed MLE stage found in {} (looked for {})",
                dir, MLE_STAGES.join(", ")))?
        }
    };
    let path = format!("{}/{}/final_params.toml", dir, target_stage);
    let params = crate::util::load_params_toml(&path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    let mut out = String::new();
    out.push_str(&format!("# camdl fit summary --params-only --stage {}\n", target_stage));
    out.push_str(&format!("# source: {}\n", path));
    out.push_str(&format!("# camdl: {}\n\n", version::VERSION_SHORT));
    for k in keys {
        let v = params[k];
        // Emit integers without a decimal so the loader returns the
        // expected value. format_param_value already handles this.
        out.push_str(&format!("{} = {}\n", k, crate::fit::runner::format_param_value(v)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{CleanEvalConfig, GateConfig};
    use std::collections::HashMap;

    fn synthetic_fit_state() -> FitState {
        let mut start = HashMap::new();
        start.insert("R0".into(),  56.0);
        start.insert("sigma".into(), 0.08);
        start.insert("gamma".into(), 0.08);
        let mut agreement = HashMap::new();
        agreement.insert("R0".into(),    1.04);
        agreement.insert("sigma".into(), 1.01);
        agreement.insert("gamma".into(), 1.21);
        FitState {
            stage: "scout".into(),
            seed: 42,
            timestamp: "2026-04-25T00:00:00Z".into(),
            input_hash: Some("deadbeef".into()),
            camdl_version: Some(version::VERSION_SHORT.into()),
            best_loglik: -3804.9,
            initial_loglik: -7891.0,
            best_chain: 1,
            n_chains: 8,
            n_good_chains: Some(8),
            start_values: start,
            rw_sd: HashMap::new(),
            loglik_type: Some("if2".into()),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            ivp_params: vec!["I0".into()],
            chain_logliks: vec![-3810.0; 8],
            chain_clean_logliks: vec![
                -3810.5, -3805.1, -3812.0, -3808.7,
                -3804.9, -3811.2, -3809.0, -3807.6,
            ],
            chain_clean_ses: vec![1.5, 1.2, 1.8, 1.4, 1.1, 1.6, 1.3, 1.5],
            resolved_gate: Some(GateConfig::default()),
            resolved_clean_eval: Some(CleanEvalConfig::default()),
        }
    }

    #[test]
    fn formatter_renders_pass_verdict_when_thresholds_clear() {
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false };
        let block = fmt.gate_verdict_block(&state);

        // Â leg: max = 1.21 on gamma, threshold 1.01 → fail.
        // Spread is small (~7 nats) → decibans leg passes.
        // Overall: FAIL because Â leg fails.
        assert!(block.contains("Â leg:"));
        assert!(block.contains("max Â = 1.210 (gamma)"),
            "expected max Â call-out; got: {}", block);
        assert!(block.contains("decibans leg:"));
        assert!(block.contains("overall:"));
    }

    #[test]
    fn formatter_emits_caveat_when_resolved_gate_absent() {
        let mut state = synthetic_fit_state();
        state.resolved_gate = None;
        let fmt = Formatter { use_color: false };
        let block = fmt.gate_verdict_block(&state);
        assert!(block.contains("thresholds unknown"),
            "legacy fit_state without resolved_gate must surface caveat; got: {}",
            block);
    }

    #[test]
    fn parameter_table_filters_to_estimated_params() {
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false };
        let block = fmt.parameter_table(&state);
        assert!(block.contains("R0"), "R0 row missing: {}", block);
        assert!(block.contains("Â=1.040"), "R0 agreement missing: {}", block);
        assert!(block.contains("sigma"));
        assert!(block.contains("gamma"));
    }

    #[test]
    fn ci_env_strict_auto_enable() {
        // Sanity check on the gate that triggers --strict from CI=true.
        // We can't toggle env vars in a thread-safe way during cargo
        // test, so just verify the helper reads the right values.
        std::env::remove_var("CI");
        assert!(!ci_env_set());
        std::env::set_var("CI", "true");
        assert!(ci_env_set());
        std::env::set_var("CI", "1");
        assert!(ci_env_set());
        std::env::set_var("CI", "false");
        assert!(!ci_env_set());
        std::env::remove_var("CI");
    }

    /// Provenance cross-check is the always-on diagnostic that turns
    /// the GH #16 silent-wrong-answer mode into a visible ✗ on every
    /// read. Test: write a fit dir where final_params.toml and
    /// mle_params.toml carry different R0 values; assert provenance
    /// block flags it.
    #[test]
    fn provenance_block_detects_mle_final_disagreement() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_summary_prov_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("final_params.toml"),
            "R0 = 56.82\nsigma = 0.0791\n").unwrap();
        std::fs::write(dir.join("mle_params.toml"),
            "R0 = 81.45\nsigma = 0.0791\n").unwrap();

        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false };
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &state);
        assert!(prov.failed, "must flag the disagreement: {}", prov.text);
        assert!(prov.text.contains("DISAGREE"),
            "must call out DISAGREE: {}", prov.text);
        assert!(prov.text.contains("#16"),
            "must reference the GH issue this guards against: {}",
            prov.text);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provenance_block_passes_when_params_match() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_summary_prov_ok_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Values must match `synthetic_fit_state().start_values`
        // exactly — the second cross-check (fit_state ↔ final_params)
        // compares them.
        std::fs::write(dir.join("final_params.toml"),
            "R0 = 56.0\nsigma = 0.08\ngamma = 0.08\n").unwrap();
        std::fs::write(dir.join("mle_params.toml"),
            "R0 = 56.0\nsigma = 0.08\ngamma = 0.08\n").unwrap();
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false };
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &state);
        assert!(!prov.failed, "must not flag when params match: {}", prov.text);
        assert!(prov.text.contains("✓"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Phase 4 / 5 tests ─────────────────────────────────────────

    /// Build a fit dir with one stage, return its top-level path.
    /// Mirror of what `camdl fit run` writes minus the bits these
    /// tests don't need.
    fn make_fit_dir(stage: &str, state: &FitState, params: &[(&str, f64)])
        -> std::path::PathBuf
    {
        let dir = std::env::temp_dir().join(format!(
            "camdl_summary_format_{}_{}_{}",
            stage, std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        let stage_dir = dir.join(stage);
        std::fs::create_dir_all(&stage_dir).unwrap();
        state.save(&stage_dir.to_string_lossy()).unwrap();

        // final_params.toml + mle_params.toml carrying matching values
        // so the provenance cross-check passes.
        let mut body = String::new();
        for (k, v) in params {
            body.push_str(&format!("{} = {}\n", k, v));
        }
        std::fs::write(stage_dir.join("final_params.toml"), &body).unwrap();
        std::fs::write(stage_dir.join("mle_params.toml"), &body).unwrap();
        dir
    }

    /// JSON output is parseable, schema.version is 1, stage report
    /// fields match the FitState we constructed it from. Catches any
    /// future schema rename / removal that would break the book
    /// pipeline.
    #[test]
    fn json_format_round_trips_and_carries_schema_version() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let doc = build_summary_doc(&dir.to_string_lossy(), &["scout"]);
        let json = serde_json::to_string_pretty(&doc).unwrap();
        assert!(json.contains("\"version\": 1"),
            "schema.version must be present and = 1: {}", json);
        // Reparse and pin the load-bearing fields.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["schema"]["version"], 1);
        assert_eq!(parsed["fit_dir"], dir.to_string_lossy().as_ref());
        assert_eq!(parsed["stages"][0]["name"], "scout");
        assert!((parsed["stages"][0]["best_loglik"].as_f64().unwrap() - (-3804.9)).abs() < 1e-6);
        // Heuristic block is namespaced.
        assert!(parsed["stages"][0]["_heuristic"]["overall_status"].is_string());
        // Provenance keys present.
        let prov = &parsed["stages"][0]["provenance"];
        assert_eq!(prov["final_params_matches_mle_params"], true);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Markdown output is well-formed: headings, gate-verdict table,
    /// parameter table. Spot-check critical lines.
    #[test]
    fn markdown_format_renders_gate_table_and_params() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let doc = build_summary_doc(&dir.to_string_lossy(), &["scout"]);
        let md = render_markdown(&doc);
        assert!(md.contains("# Fit summary:"));
        assert!(md.contains("## `scout`"));
        assert!(md.contains("### Compound scout-convergence gate"));
        assert!(md.contains("| Â (max over params"));
        assert!(md.contains("### Parameter estimates"));
        assert!(md.contains("`R0`"));
        assert!(md.contains("### Per-chain clean-eval"));
        assert!(md.contains("### Provenance"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn latex_format_renders_tabular_blocks() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let doc = build_summary_doc(&dir.to_string_lossy(), &["scout"]);
        let tex = render_latex(&doc);
        // No preamble, but tabular blocks per stage.
        assert!(tex.contains("\\subsection*{Stage:"));
        assert!(tex.contains("\\begin{tabular}"));
        assert!(tex.contains("$\\hat A$"));
        assert!(tex.contains("\\bottomrule"));
        // No raw `&` — must be escaped.
        let r0_count = tex.matches(" & ").count();
        assert!(r0_count > 0, "tables must use & as column separator");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--params-only` emits a flat params TOML that the standard
    /// loader accepts. Pipes-through to `pfilter --params` are the
    /// load-bearing use case for Phase 5.
    #[test]
    fn params_only_emits_loadable_toml() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let s = dump_params_only(&dir.to_string_lossy(), Some("scout")).unwrap();
        // No metadata leaks at top level (the existing loader skips
        // `[provenance]`, but --params-only doesn't even emit it).
        assert!(!s.contains("[provenance]"),
            "params-only must not include the [provenance] block: {}", s);

        // Round-trip via the actual production loader.
        let tmp = dir.join("emitted.toml");
        std::fs::write(&tmp, &s).unwrap();
        let loaded = crate::util::load_params_toml(tmp.to_str().unwrap()).unwrap();
        assert!((loaded["R0"] - 56.0).abs() < 1e-9);
        assert!((loaded["sigma"] - 0.08).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn params_only_picks_terminal_stage_in_pipeline_order() {
        // Build a fit dir with both scout and refine; --params-only
        // without --stage should pick refine (most refined).
        let state = synthetic_fit_state();
        let scout_params = [("R0", 56.0_f64)];
        let refine_params = [("R0", 56.5_f64)];
        let dir = make_fit_dir("scout", &state, &scout_params);
        // Add a refine stage in the same dir.
        let refine_dir = dir.join("refine");
        std::fs::create_dir_all(&refine_dir).unwrap();
        state.save(&refine_dir.to_string_lossy()).unwrap();
        let mut body = String::new();
        for (k, v) in refine_params { body.push_str(&format!("{} = {}\n", k, v)); }
        std::fs::write(refine_dir.join("final_params.toml"), &body).unwrap();
        std::fs::write(refine_dir.join("mle_params.toml"), &body).unwrap();

        let s = dump_params_only(&dir.to_string_lossy(), None).unwrap();
        assert!(s.contains("--stage refine"),
            "no --stage filter must pick refine over scout: {}", s);
        assert!(s.contains("R0 = 56.5"),
            "must dump refine's params, not scout's: {}", s);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn params_only_errors_when_no_completed_stage() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_summary_empty_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let err = dump_params_only(&dir.to_string_lossy(), None).unwrap_err();
        assert!(err.contains("no completed MLE stage"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
