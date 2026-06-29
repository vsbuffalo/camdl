//! `camdl compare` — multi-model prequential comparison table.
//!
//! Consumes `prequential.json` artifacts written by `fit run` (PFilter
//! stage) or `camdl pfilter --save-prequential`. Computes Δelpd / Δcrps
//! with paired pointwise SEs, refuses structurally unfair comparisons
//! (T_score mismatch by default), renders as table / markdown / JSON.
//!
//! See docs/dev/proposals/2026-04-20-prequential-evaluation.md §8.
//!
//! Scope (Part I):
//!   - baseline-centered Δelpd + paired SE
//!   - Δcrps + PIT 90% coverage column
//!   - T_score fairness preflight (override: --allow-mismatched-horizon)
//!   - formats: table (default), md, json
//!   - compare.toml for reproducible multi-model specs
//! Out of scope (Part II): betting mode, CAS ref resolution, data_hash /
//!   obs-model / backend preflights, anti-pattern detection beyond T_score,
//!   stacking, plotting.

use crate::fit::handle::ResolvedFit;
use serde::Deserialize;
use sim::inference::prequential::PrequentialTrace;
use std::path::{Path, PathBuf};

/// Default particle count for auto-deriving a prequential from a fit handle.
/// Applied uniformly across the compared fits (see [`DeriveSettings`]).
pub(crate) const DEFAULT_DERIVE_PARTICLES: usize = 1000;

/// Default filter seed for auto-deriving a prequential from a fit handle.
pub(crate) const DEFAULT_DERIVE_SEED: u64 = 1;

/// Settings applied uniformly to every fit handle whose prequential is
/// auto-derived, so T_score and the scores stay commensurable across the
/// compared fits. An explicit `prequential.json` input ignores these (it is
/// read as-is, at whatever particles/seed produced it).
#[derive(Debug, Clone, Copy)]
struct DeriveSettings {
    particles: usize,
    seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)] // gh#241 G3: reject typo'd compare.toml keys instead of silently dropping
struct CompareToml {
    baseline: Option<String>,
    metrics: Option<Vec<String>>,
    format: Option<String>,
    #[serde(rename = "model")]
    models: Vec<CompareModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareModelEntry {
    name: String,
    path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format { Table, Md, Json }

/// Per-model row in the comparison table.
struct Row {
    name: String,
    path: String,
    trace: PrequentialTrace,
}

pub fn cmd_compare(a: &crate::args::CompareArgs) {
    let config_path: Option<String> = a.config.clone();
    let baseline: Option<String> = a.baseline.clone();
    let allow_mismatched_horizon = a.allow_mismatched_horizon;
    let positional: Vec<String> = a.paths.clone();
    let derive = DeriveSettings { particles: a.particles, seed: a.seed };
    let metrics_cli: Option<Vec<String>> = a.metrics.as_ref().map(|s|
        s.split(',').map(|t| t.trim().to_string()).collect());
    let format = match a.format.as_str() {
        "table" => Format::Table,
        "md"    => Format::Md,
        "json"  => Format::Json,
        other   => {
            eprintln!("error: --format must be table|md|json (got '{}')", other);
            std::process::exit(1);
        }
    };

    // Resolve model list: CLI positional > compare.toml
    let (models, cfg_baseline, cfg_metrics, cfg_format) = if let Some(path) = config_path {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: cannot read {}: {}", path, e);
            std::process::exit(1);
        });
        let cfg: CompareToml = toml::from_str(&text).unwrap_or_else(|e| {
            eprintln!("error: parsing {}: {}", path, e);
            std::process::exit(1);
        });
        let fmt = cfg.format.as_deref().and_then(|f| match f {
            "table" => Some(Format::Table),
            "md"    => Some(Format::Md),
            "json"  => Some(Format::Json),
            _       => None,
        });
        (cfg.models, cfg.baseline, cfg.metrics, fmt)
    } else if positional.len() >= 2 {
        let models = positional.iter().map(|p| CompareModelEntry {
            name: std::path::Path::new(p).file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.clone()),
            path: p.clone(),
        }).collect();
        (models, None, None, None)
    } else {
        eprintln!("error: compare requires either --config FILE or ≥2 fit stage paths");
        std::process::exit(1);
    };

    if models.len() < 2 {
        eprintln!("error: compare needs ≥2 models; got {}", models.len());
        std::process::exit(1);
    }

    // Load traces.
    let rows: Vec<Row> = models.into_iter().map(|m| {
        let trace = load_trace(&m.path, derive).unwrap_or_else(|e| {
            eprintln!("error loading trace for '{}' at '{}': {}", m.name, m.path, e);
            std::process::exit(1);
        });
        Row { name: m.name, path: m.path, trace }
    }).collect();

    // Fairness: T_score (n_scored) must agree across rows unless overridden.
    let t_scores: Vec<usize> = rows.iter().map(|r| r.trace.n_scored()).collect();
    let t_ref = t_scores[0];
    let t_mismatch = t_scores.iter().any(|&t| t != t_ref);
    if t_mismatch && !allow_mismatched_horizon {
        eprintln!("error: T_score differs across models: {:?}", t_scores);
        eprintln!("       Δelpd and Δcrps are not commensurable.");
        eprintln!("       Pass --allow-mismatched-horizon to render (uncomparable Δ columns → '—').");
        std::process::exit(2);
    }

    // #295 Ask 1: surface the optimism so an in-sample / plug-in score is never
    // silently read as an honest out-of-sample forecast score. Today every trace
    // is plug-in + in-sample, so this always fires; when posterior / LFO traces
    // exist, the per-trace tag drives it.
    if let Some(caveat) = rows.iter().find_map(|r| r.trace.optimism_caveat()) {
        eprintln!("note: {caveat}");
    }

    // Baseline: explicit > cfg > argmax elpd.
    let baseline_name = baseline.or(cfg_baseline).unwrap_or_else(|| {
        let best = rows.iter()
            .max_by(|a, b| a.trace.elpd().partial_cmp(&b.trace.elpd())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        best.name.clone()
    });
    let base_idx = rows.iter().position(|r| r.name == baseline_name)
        .unwrap_or_else(|| {
            eprintln!("error: baseline '{}' not found among models: {:?}",
                baseline_name,
                rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>());
            std::process::exit(1);
        });

    let metrics_chosen = metrics_cli.or(cfg_metrics)
        .unwrap_or_else(|| vec!["elpd".into(), "crps".into(), "pit_cov90".into()]);
    let fmt_final = cfg_format.unwrap_or(format);

    match fmt_final {
        Format::Json  => render_json(&rows, base_idx, &metrics_chosen),
        Format::Md    => render_md(&rows, base_idx, &metrics_chosen, t_mismatch),
        Format::Table => render_table(&rows, base_idx, &metrics_chosen, t_mismatch),
    }
}

/// Resolve a single comparison input to a `PrequentialTrace`. Two paths:
///
/// 1. **Explicit prequential** (kept, tried first): a `.json` file that exists,
///    or a directory holding `prequential.json` — read+parsed as-is. Preserves
///    `pfilter --save-prequential` and stage-dir inputs; the derive settings do
///    not touch it.
/// 2. **Fit handle** (Phase 2a): `@label` / hash prefix / run dir / `fit.toml`.
///    The prequential is DERIVED by invoking the canonical `camdl pfilter` at
///    the fit's sealed θ̂ — never by reimplementing the filter (the obs-model
///    assembly is already triplicated; a fourth copy is forbidden).
fn load_trace(path: &str, derive: DeriveSettings) -> Result<PrequentialTrace, String> {
    if let Some(trace) = try_load_explicit_prequential(path)? {
        return Ok(trace);
    }
    // Not an explicit prequential → treat as a fit handle and derive.
    match crate::fit::handle::resolve_fit(path) {
        Ok(resolved) => derive_prequential(&resolved, derive),
        Err(resolve_err) => Err(format!(
            "'{path}' is neither a prequential trace nor a resolvable fit handle.\n  \
             - as a fit handle: {resolve_err}\n  \
             - as a prequential path: no prequential.json at '{path}' (or \
             '{path}/prequential.json') — run `camdl pfilter --save-prequential` \
             or `camdl fit run` with a pfilter stage to generate one."
        )),
    }
}

/// Try to read `path` as an explicit prequential artifact. Returns `Ok(None)`
/// (fall through to fit-handle derivation) when it is not one: a non-existent
/// path, a `@label` / hash prefix, a `.toml`, or a directory with no
/// `prequential.json`. Only a `.json` file or a `<dir>/prequential.json` is
/// read — so a `fit.toml` handle is never mis-parsed as a trace.
fn try_load_explicit_prequential(path: &str) -> Result<Option<PrequentialTrace>, String> {
    let p = Path::new(path);
    let json_path = if p.is_dir() {
        let jp = p.join("prequential.json");
        if !jp.exists() {
            return Ok(None);
        }
        jp
    } else if p.extension().map(|e| e.eq_ignore_ascii_case("json")).unwrap_or(false) {
        if !p.exists() {
            return Ok(None);
        }
        p.to_path_buf()
    } else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("{}: {}", json_path.display(), e))?;
    let trace = serde_json::from_str::<PrequentialTrace>(&text)
        .map_err(|e| format!("parsing {}: {}", json_path.display(), e))?;
    Ok(Some(trace))
}

/// Derive a prequential trace from a sealed fit by invoking the canonical
/// `camdl pfilter` at the fit's winning θ̂. This routes through the one
/// production filter + obs-model assembly rather than rebuilding it, so a
/// derived trace can never silently diverge from a hand-run `pfilter`.
///
/// Steps: (a) θ̂ from the winning stage as a flat params TOML; (b) the model
/// (archived IR if present, else the loose source the config names); (c) the
/// fit's data streams as `--data NAME=PATH`; (d) run `pfilter
/// --save-prequential` to a temp stem; (e) read back the `{stem}.json` trace.
/// All temp files live in the system temp dir and are best-effort cleaned up.
fn derive_prequential(resolved: &ResolvedFit, derive: DeriveSettings) -> Result<PrequentialTrace, String> {
    let segment = &resolved.segment;
    let config = &resolved.config;

    // (a) θ̂ — the plug-in point the prequential is scored at. Routed through
    // the draws-cloud authority (`resolve_posterior_draws`), NOT a per-method
    // point-estimate file: a Bayesian fit (PGAS/PMMH/MH) writes no
    // `final_params.toml` — its θ̂ is the posterior MEAN over `draws.tsv`; only
    // an optimizer fit (IF2/NLopt) has a single winner file. The headline
    // `compare @pgas_a @pgas_b` workflow used to dead-end on the missing file.
    let params_toml = point_estimate_params_toml(segment)?;

    // (b) model — prefer the self-contained archived IR (Phase 1a), else the
    // loose source the config names (recompiled by pfilter). config.model.camdl
    // is already absolute (resolved against the fit.toml dir at load).
    let archived = segment.join("model.ir.json");
    let model_path: PathBuf = if archived.is_file() {
        archived
    } else {
        PathBuf::from(&config.model.camdl)
    };

    // (c) data — stream → absolute path. The config's data paths were made
    // absolute (relative to the original fit.toml) at load, so they pass to the
    // child unchanged.
    let data = config.data.as_ref().ok_or_else(|| format!(
        "fit at {} has no [data] block — there are no observations to score, so a \
         prequential trace cannot be derived.\n  Provide an explicit \
         prequential.json, or compare fits that bind data.",
        segment.display()))?;
    let streams = resolve_streams(data, &model_path)?;
    if streams.is_empty() {
        return Err(format!(
            "fit at {} has a [data] block with no observation streams.",
            segment.display()));
    }

    // Temp files: unique per process + fit segment so concurrent compares don't
    // collide. STEM has no `.` so pfilter's `{stem}.json` / `{stem}.tsv` are
    // unambiguous.
    let seg_slug = segment
        .file_name()
        .map(|s| s.to_string_lossy().replace(|c: char| !c.is_ascii_alphanumeric(), "_"))
        .unwrap_or_else(|| "fit".into());
    let base = std::env::temp_dir()
        .join(format!("camdl_compare_{}_{}", std::process::id(), seg_slug));
    let theta_path = base.with_extension("theta.toml");
    let preq_stem = base.to_string_lossy().into_owned() + "_preq";
    let preq_json = format!("{preq_stem}.json");
    let preq_tsv = format!("{preq_stem}.tsv");
    let cleanup = || {
        let _ = std::fs::remove_file(&theta_path);
        let _ = std::fs::remove_file(&preq_json);
        let _ = std::fs::remove_file(&preq_tsv);
    };

    if let Err(e) = std::fs::write(&theta_path, &params_toml) {
        cleanup();
        return Err(format!("writing temp params {}: {}", theta_path.display(), e));
    }

    // (d) run the canonical filter.
    let exe = std::env::current_exe()
        .map_err(|e| { cleanup(); format!("cannot locate the running camdl binary: {e}") })?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("pfilter")
        .arg(&model_path)
        .arg("--params").arg(&theta_path)
        .arg("--save-prequential").arg(&preq_stem)
        .arg("--particles").arg(derive.particles.to_string())
        .arg("--seed").arg(derive.seed.to_string())
        .env("CAMDL_SKIP_VERSION_CHECK", "1");
    for (name, abs_path) in &streams {
        cmd.arg("--data").arg(format!("{name}={abs_path}"));
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => { cleanup(); return Err(format!("spawning `camdl pfilter`: {e}")); }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".into());
        cleanup();
        return Err(format!(
            "deriving prequential via `camdl pfilter` failed (exit {code}) for fit {}:\n{}",
            segment.display(),
            stderr.trim()));
    }

    // (e) read back the trace.
    let text = match std::fs::read_to_string(&preq_json) {
        Ok(t) => t,
        Err(e) => { cleanup(); return Err(format!(
            "`camdl pfilter` succeeded but its prequential output {preq_json} could not be read: {e}")); }
    };
    let trace = serde_json::from_str::<PrequentialTrace>(&text)
        .map_err(|e| format!("parsing derived prequential {preq_json}: {e}"));
    cleanup();
    trace
}

/// θ̂ for the prequential as a flat params TOML, routed through the draws-cloud
/// authority. A Bayesian fit (its terminal stage wrote a `draws.tsv`) plugs in
/// the posterior MEAN over the cloud; an optimizer fit (no cloud) plugs in its
/// winner file (`final_params.toml`, via `winner_params_toml`). This is the
/// "resolve by artifact, not by method name" rule — the headline `compare
/// @pgas_a @pgas_b` Bayesian comparison was previously dead-ending on the
/// IF2-only `final_params.toml`.
fn point_estimate_params_toml(segment: &Path) -> Result<String, String> {
    match crate::posterior_draws::resolve_posterior_draws(&segment.to_string_lossy(), None) {
        Ok(pdraws) => posterior_mean_params_toml(&pdraws.draws_path),
        // No posterior cloud → an optimizer fit; its θ̂ is the single winner.
        Err(_) => crate::fit::fit_summary::winner_params_toml(segment, None),
    }
}

/// The posterior MEAN of every column in a `draws.tsv` as a flat params TOML —
/// the plug-in point a prequential is scored at for a Bayesian fit. Every model
/// parameter is a column (estimated + fixed); a fixed column is constant, so its
/// mean is the fixed value. Columns are emitted in sorted order for determinism.
fn posterior_mean_params_toml(draws_path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(draws_path)
        .map_err(|e| format!("reading {}: {e}", draws_path.display()))?;
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| format!("empty draws.tsv at {}", draws_path.display()))?
        .split('\t')
        .map(|s| s.to_string())
        .collect();
    let mut sums = vec![0.0f64; header.len()];
    let mut n = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != header.len() {
            return Err(format!("ragged row in {} ({} cols, header has {})",
                draws_path.display(), fields.len(), header.len()));
        }
        for (i, f) in fields.iter().enumerate() {
            sums[i] += f.parse::<f64>().map_err(|_| {
                format!("non-numeric draw '{f}' in {}", draws_path.display())
            })?;
        }
        n += 1;
    }
    if n == 0 {
        return Err(format!("no posterior draws in {}", draws_path.display()));
    }
    let mut idx: Vec<usize> = (0..header.len()).collect();
    idx.sort_by(|&a, &b| header[a].cmp(&header[b]));
    let mut out = String::new();
    out.push_str("# camdl compare: posterior-mean point estimate (θ̂) for the prequential\n");
    out.push_str(&format!("# source: {} ({n} draws)\n\n", draws_path.display()));
    for i in idx {
        // gh#322: `chain` / `draw` are posterior key columns, not parameters —
        // never emit them into θ̂ (a `chain = …` line would make `pfilter` reject
        // an unknown parameter). This parser reads the file directly rather than
        // via the shared loader, so it strips them itself.
        if header[i] == "chain" || header[i] == "draw" {
            continue;
        }
        out.push_str(&format!(
            "{} = {}\n",
            header[i],
            crate::fit::runner::format_param_value(sums[i] / n as f64)
        ));
    }
    Ok(out)
}

/// Resolve a fit's `[data]` spec to a stream-name → absolute-path map for
/// `--data NAME=PATH`. The per-stream `observations` map is used directly; the
/// single-file shorthand (`file = "..."`) is expanded via the existing
/// [`DataSpec::effective_observations`] seam, which needs the model's declared
/// observation-stream names (loaded from `model_path`).
fn resolve_streams(
    data: &crate::fit::config_v2::DataSpec,
    model_path: &Path,
) -> Result<indexmap::IndexMap<String, String>, String> {
    if !data.observations.is_empty() {
        return Ok(data.observations.clone());
    }
    if data.file.is_some() {
        let (model, _) = crate::util::load_model(&model_path.to_string_lossy())
            .map_err(|e| format!(
                "loading model {} to expand the single-file [data] shorthand: {e}",
                model_path.display()))?;
        let names: Vec<String> = model.observations.iter().map(|o| o.name.clone()).collect();
        return data.effective_observations(&names);
    }
    Err("the fit's [data] block has neither `observations` nor `file`".to_string())
}

/// Paired Δ = sum_t (a_t − b_t); paired SE = sqrt(T · Var_t(a_t − b_t)).
/// Returns (delta, se) or (NaN, NaN) if horizons mismatch.
fn paired_delta(a: &PrequentialTrace, b: &PrequentialTrace, field: Field)
    -> (f64, f64)
{
    if a.n_scored() != b.n_scored() || a.n_scored() == 0 {
        return (f64::NAN, f64::NAN);
    }
    let diffs: Vec<f64> = a.steps.iter().zip(&b.steps)
        .map(|(x, y)| match field {
            Field::LogScore => x.log_score - y.log_score,
            Field::Crps     => x.crps - y.crps,
        })
        .collect();
    let t = diffs.len() as f64;
    let delta: f64 = diffs.iter().sum();
    let mean = delta / t;
    let var = if t > 1.0 {
        diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (t - 1.0)
    } else { 0.0 };
    let se = (t * var).sqrt();
    (delta, se)
}

#[derive(Copy, Clone)]
enum Field { LogScore, Crps }

/// Format an e-value for display. exp(Δelpd) ranges over many orders of
/// magnitude — compact decimal for the "interesting" band [0.001, 1000]
/// and scientific notation outside. E_T = 1 means "tied with baseline";
/// E_T = 100 means "100× more likely than baseline under its own predictive";
/// E_T = 0.01 means "1/100× as likely."
fn fmt_e_value(e: f64) -> String {
    if !e.is_finite() { return "—".into(); }
    if e == 0.0 { return "0".into(); }
    if e >= 1000.0 || e < 0.001 {
        format!("{:.2e}", e)
    } else {
        format!("{:.3}", e)
    }
}

fn render_table(rows: &[Row], base_idx: usize, metrics: &[String], t_mismatch: bool) {
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");

    // Δelpd (nats) is the primary machine-readable column; "evidence"
    // (decibans + Jeffreys label) is the human-interpretable alongside —
    // see docs/dev/proposals/2026-04-23-evidence-in-decibans.md §Scope.
    let mut header = vec!["Model".to_string(), "T_score".into(), "elpd".into(),
        "Δelpd".into(), "E_T".into(), "se(Δ)".into(), "evidence".into()];
    if want_crps { header.push("crps".into()); header.push("Δcrps".into()); }
    if want_pit  { header.push("PIT_cov90".into()); }

    let base = &rows[base_idx].trace;
    // Render order: ascending Δelpd (worst → best). Baseline drops in
    // at its natural Δelpd = 0 slot. Best candidate lands at the
    // bottom — reader's eye sees the recommended model last. When
    // T_score mismatches block per-row deltas, we can't sort
    // meaningfully, so leave the input order alone in that path.
    let order: Vec<usize> = if t_mismatch {
        (0..rows.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| {
            let da = if a == base_idx { 0.0 }
                else { paired_delta(&rows[a].trace, base, Field::LogScore).0 };
            let db = if b == base_idx { 0.0 }
                else { paired_delta(&rows[b].trace, base, Field::LogScore).0 };
            // NaN delta → place row at the top (worst-case ranking).
            match (da.is_nan(), db.is_nan()) {
                (true, true)  => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        idx
    };
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for &i in &order {
        let r = &rows[i];
        let elpd = r.trace.elpd();
        let mut row = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", elpd),
        ];
        if i == base_idx {
            row.push("—".into());   // Δelpd
            row.push("—".into());   // E_T
            row.push("—".into());   // se(Δ)
            row.push("—".into());   // evidence (dB + Jeffreys label)
        } else if t_mismatch {
            row.push("—".into());
            row.push("—".into());
            row.push("—".into());
            row.push("—".into());
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            row.push(format!("{:+.2}", d));
            row.push(fmt_e_value(d.exp()));
            row.push(format!("{:.2}", se));
            let (_, evidence) = crate::evidence::evidence_cells(d);
            row.push(evidence);
        }
        if want_crps {
            row.push(format!("{:.3}", r.trace.mean_crps()));
            if i == base_idx || t_mismatch {
                row.push("—".into());
            } else {
                let (d, _) = paired_delta(&r.trace, base, Field::Crps);
                let mean_diff = d / r.trace.n_scored() as f64;
                row.push(format!("{:+.3}", mean_diff));
            }
        }
        if want_pit {
            row.push(format!("{:.2}", r.trace.pit_coverage(0.90)));
        }
        body.push(row);
    }

    let widths: Vec<usize> = (0..header.len()).map(|c| {
        let h = header[c].chars().count();
        let b = body.iter().map(|r| r[c].chars().count()).max().unwrap_or(0);
        h.max(b)
    }).collect();

    let sep = |cols: &[usize]| -> String {
        let total: usize = cols.iter().sum::<usize>() + 3 * (cols.len() - 1);
        "─".repeat(total)
    };

    print_row(&header, &widths);
    println!("{}", sep(&widths));
    for row in &body {
        print_row(row, &widths);
    }

    println!();
    println!("Scored steps: {} (t0={}).  Baseline: {}.",
        base.n_scored(), base.t0, rows[base_idx].name);
    if !t_mismatch {
        println!("Sorted by Δelpd ascending — best-supported model at the bottom.");
    }
    if t_mismatch {
        println!("⚠ T_score differs across models — Δ columns suppressed \
            (--allow-mismatched-horizon was set).");
    }
    // PIT warnings — flag clear miscalibration.
    for r in rows {
        let cov = r.trace.pit_coverage(0.90);
        if cov < 0.70 {
            println!("⚠ {}: PIT 90%-coverage {:.2} (nominal 0.90) — likely overconfident.",
                r.name, cov);
        }
    }
    // Propagate trace-level warnings.
    for r in rows {
        for w in &r.trace.warnings {
            println!("ⓘ {}: {:?}", r.name, w);
        }
    }
}

fn print_row(cells: &[String], widths: &[usize]) {
    let parts: Vec<String> = cells.iter().zip(widths)
        .map(|(c, w)| format!("{:>width$}", c, width = w))
        .collect();
    // Left-align the first column (model name) for readability.
    let mut out = String::new();
    for (i, (c, w)) in cells.iter().zip(widths).enumerate() {
        if i == 0 {
            out.push_str(&format!("{:<width$}", c, width = w));
        } else {
            out.push_str("   ");
            out.push_str(&parts[i]);
        }
    }
    println!("{}", out);
}

fn render_md(rows: &[Row], base_idx: usize, metrics: &[String], t_mismatch: bool) {
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");

    let mut header = vec!["Model", "T_score", "elpd", "Δelpd", "E_T", "se(Δ)", "evidence"];
    if want_crps { header.push("crps"); header.push("Δcrps"); }
    if want_pit  { header.push("PIT_cov90"); }
    println!("| {} |", header.join(" | "));
    println!("|{}|", header.iter().map(|_| "---").collect::<Vec<_>>().join("|"));

    let base = &rows[base_idx].trace;
    // Same render order as the table renderer: ascending Δelpd
    // (best-supported last). When T_score mismatches block per-row
    // deltas, leave input order alone.
    let order: Vec<usize> = if t_mismatch {
        (0..rows.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| {
            let da = if a == base_idx { 0.0 }
                else { paired_delta(&rows[a].trace, base, Field::LogScore).0 };
            let db = if b == base_idx { 0.0 }
                else { paired_delta(&rows[b].trace, base, Field::LogScore).0 };
            match (da.is_nan(), db.is_nan()) {
                (true, true)  => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        idx
    };
    for &i in &order {
        let r = &rows[i];
        let mut cells: Vec<String> = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", r.trace.elpd()),
        ];
        if i == base_idx || t_mismatch {
            cells.push("—".into());  // Δelpd
            cells.push("—".into());  // E_T
            cells.push("—".into());  // se(Δ)
            cells.push("—".into());  // evidence
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            cells.push(format!("{:+.2}", d));
            cells.push(fmt_e_value(d.exp()));
            cells.push(format!("{:.2}", se));
            let (_, evidence) = crate::evidence::evidence_cells(d);
            cells.push(evidence);
        }
        if want_crps {
            cells.push(format!("{:.3}", r.trace.mean_crps()));
            if i == base_idx || t_mismatch {
                cells.push("—".into());
            } else {
                let (d, _) = paired_delta(&r.trace, base, Field::Crps);
                let mean_diff = d / r.trace.n_scored() as f64;
                cells.push(format!("{:+.3}", mean_diff));
            }
        }
        if want_pit { cells.push(format!("{:.2}", r.trace.pit_coverage(0.90))); }
        println!("| {} |", cells.join(" | "));
    }
    if !t_mismatch {
        println!();
        println!("_Sorted by Δelpd ascending — best-supported model at the bottom._");
    }
}

fn render_json(rows: &[Row], base_idx: usize, metrics: &[String]) {
    use serde_json::json;
    let base = &rows[base_idx].trace;
    let entries: Vec<serde_json::Value> = rows.iter().enumerate().map(|(i, r)| {
        let (d_elpd, se_elpd) = if i == base_idx { (f64::NAN, f64::NAN) }
            else { paired_delta(&r.trace, base, Field::LogScore) };
        let (d_crps, _) = if i == base_idx { (f64::NAN, f64::NAN) }
            else { paired_delta(&r.trace, base, Field::Crps) };
        let mean_dcrps = if r.trace.n_scored() == 0 { f64::NAN }
            else { d_crps / r.trace.n_scored() as f64 };
        let e_t = if d_elpd.is_finite() { d_elpd.exp() } else { f64::NAN };
        // Evidence: Δelpd (nats) → decibans + Jeffreys label. Derived
        // field for human-interpretable consumption; nats remain the
        // primary machine-readable quantity (delta_elpd). See
        // docs/dev/proposals/2026-04-23-evidence-in-decibans.md.
        let (d_elpd_db, evidence_label) = if d_elpd.is_finite() {
            let db = d_elpd * crate::evidence::NATS_TO_DB;
            (option_finite(db), serde_json::json!(crate::evidence::jeffreys_label(db)))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
        json!({
            "name": r.name,
            "path": r.path,
            "t_score": r.trace.n_scored(),
            "elpd": r.trace.elpd(),
            "delta_elpd": option_finite(d_elpd),
            "delta_elpd_db": d_elpd_db,
            "evidence_label": evidence_label,
            "e_t": option_finite(e_t),
            "se_delta_elpd": option_finite(se_elpd),
            "mean_crps": r.trace.mean_crps(),
            "delta_mean_crps": option_finite(mean_dcrps),
            "pit_cov90": r.trace.pit_coverage(0.90),
        })
    }).collect();
    let out = json!({
        "baseline": rows[base_idx].name,
        "metrics": metrics,
        "rows": entries,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

fn option_finite(x: f64) -> serde_json::Value {
    if x.is_finite() { serde_json::json!(x) } else { serde_json::Value::Null }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#241 G3: `deny_unknown_fields` — a typo'd compare.toml key must ERROR.
    #[test]
    fn compare_toml_rejects_unknown_keys() {
        let ok = "baseline = \"a\"\n[[model]]\nname = \"a\"\npath = \"p\"\n";
        assert!(toml::from_str::<CompareToml>(ok).is_ok(), "valid compare.toml must parse");

        let bad_top = "baselne = \"a\"\n[[model]]\nname = \"a\"\npath = \"p\"\n"; // typo: baselne
        assert!(
            toml::from_str::<CompareToml>(bad_top).is_err(),
            "a typo'd top-level key must be rejected"
        );

        let bad_model = "[[model]]\nname = \"a\"\npath = \"p\"\nlabel = \"x\"\n"; // unknown: label
        assert!(
            toml::from_str::<CompareToml>(bad_model).is_err(),
            "an unknown [[model]] key must be rejected"
        );
    }
}

