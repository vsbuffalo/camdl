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

use serde::Deserialize;
use sim::inference::prequential::PrequentialTrace;

#[derive(Debug, Clone, Deserialize)]
struct CompareToml {
    baseline: Option<String>,
    metrics: Option<Vec<String>>,
    format: Option<String>,
    #[serde(rename = "model")]
    models: Vec<CompareModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
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

pub fn cmd_compare(args: &[String]) {
    let mut config_path: Option<String> = None;
    let mut baseline: Option<String> = None;
    let mut format = Format::Table;
    let mut allow_mismatched_horizon = false;
    let mut positional: Vec<String> = Vec::new();
    let mut metrics_cli: Option<Vec<String>> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config"    => { i += 1; config_path = Some(args[i].clone()); }
            "--baseline"  => { i += 1; baseline = Some(args[i].clone()); }
            "--format"    => {
                i += 1;
                format = match args[i].as_str() {
                    "table" => Format::Table,
                    "md"    => Format::Md,
                    "json"  => Format::Json,
                    other   => {
                        eprintln!("error: --format must be table|md|json (got '{}')", other);
                        std::process::exit(1);
                    }
                };
            }
            "--metric" | "--metrics" => {
                i += 1;
                metrics_cli = Some(args[i].split(',').map(|s| s.trim().to_string()).collect());
            }
            "--allow-mismatched-horizon" => { allow_mismatched_horizon = true; }
            "--help" | "-h" => { compare_help(); }
            s if s.starts_with("--") => {
                eprintln!("error: unknown flag '{}'", s);
                compare_help();
            }
            other => { positional.push(other.to_string()); }
        }
        i += 1;
    }

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
        compare_help();
    };

    if models.len() < 2 {
        eprintln!("error: compare needs ≥2 models; got {}", models.len());
        std::process::exit(1);
    }

    // Load traces.
    let rows: Vec<Row> = models.into_iter().map(|m| {
        let trace = load_trace(&m.path).unwrap_or_else(|e| {
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

fn load_trace(path: &str) -> Result<PrequentialTrace, String> {
    let p = std::path::Path::new(path);
    // Accept either a path to prequential.json directly or a stage dir
    // that contains prequential.json.
    let json_path = if p.is_dir() {
        p.join("prequential.json")
    } else {
        p.to_path_buf()
    };
    if !json_path.exists() {
        return Err(format!(
            "no prequential.json at '{}' — run `camdl pfilter --save-prequential` or \
             `camdl fit run` with a pfilter stage to generate one.",
            json_path.display()));
    }
    let text = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("{}: {}", json_path.display(), e))?;
    serde_json::from_str::<PrequentialTrace>(&text)
        .map_err(|e| format!("parsing {}: {}", json_path.display(), e))
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

fn render_table(rows: &[Row], base_idx: usize, metrics: &[String], t_mismatch: bool) {
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");

    let mut header = vec!["Model".to_string(), "T_score".into(), "elpd".into(),
        "Δelpd".into(), "se(Δ)".into()];
    if want_crps { header.push("crps".into()); header.push("Δcrps".into()); }
    if want_pit  { header.push("PIT_cov90".into()); }

    let base = &rows[base_idx].trace;
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        let elpd = r.trace.elpd();
        let mut row = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", elpd),
        ];
        if i == base_idx {
            row.push("—".into());
            row.push("—".into());
        } else if t_mismatch {
            row.push("—".into());
            row.push("—".into());
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            row.push(format!("{:+.2}", d));
            row.push(format!("{:.2}", se));
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

    let mut header = vec!["Model", "T_score", "elpd", "Δelpd", "se(Δ)"];
    if want_crps { header.push("crps"); header.push("Δcrps"); }
    if want_pit  { header.push("PIT_cov90"); }
    println!("| {} |", header.join(" | "));
    println!("|{}|", header.iter().map(|_| "---").collect::<Vec<_>>().join("|"));

    let base = &rows[base_idx].trace;
    for (i, r) in rows.iter().enumerate() {
        let mut cells: Vec<String> = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", r.trace.elpd()),
        ];
        if i == base_idx || t_mismatch {
            cells.push("—".into()); cells.push("—".into());
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            cells.push(format!("{:+.2}", d)); cells.push(format!("{:.2}", se));
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
        json!({
            "name": r.name,
            "path": r.path,
            "t_score": r.trace.n_scored(),
            "elpd": r.trace.elpd(),
            "delta_elpd": option_finite(d_elpd),
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

fn compare_help() -> ! {
    eprintln!("camdl compare — multi-model prequential comparison");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  camdl compare PATH1 PATH2 [PATH3 ...] [OPTIONS]");
    eprintln!("  camdl compare --config compare.toml [OPTIONS]");
    eprintln!();
    eprintln!("Each PATH points to a stage directory containing prequential.json");
    eprintln!("(written by `fit run` pfilter stages or `pfilter --save-prequential`).");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --config FILE             compare.toml with [[model]] entries");
    eprintln!("  --baseline NAME           Reference for Δ columns (default: argmax elpd)");
    eprintln!("  --metric elpd,crps,pit    Metrics to display (default: all three)");
    eprintln!("  --format table|md|json    Output format (default: table)");
    eprintln!("  --allow-mismatched-horizon");
    eprintln!("                            Render even if T_score differs across models");
    eprintln!("                            (Δ columns → '—').");
    eprintln!();
    eprintln!("See docs/dev/proposals/2026-04-20-prequential-evaluation.md §8.");
    std::process::exit(0);
}
