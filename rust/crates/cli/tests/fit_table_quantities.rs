//! End-to-end acceptance for `camdl fit table --quantity <NAME>` — Phase 2b of
//! the predictive-ergonomics work. A `fit table` column surfaces the posterior
//! median (q50) of a scalar generated quantity per fit, read from
//! `<fit_dir>/quantities/<NAME>.tsv` or DERIVED on demand by spawning
//! `fit predict --horizon free_forward` for fits that carry a posterior cloud.
//!
//! Properties under test:
//!   (a) derive-on-demand: a PGAS fit that was NOT predicted gets its `peak`
//!       column filled (the table ran predict), and the derived TSV now exists;
//!   (b) read-existing: a subsequent table read returns the same q50 the derive
//!       wrote into `quantities/peak.tsv`;
//!   (c) an IF2 (optimizer) fit renders `—` — predict refuses a point-estimate
//!       fit, and the table doesn't crash;
//!   (d) the default `fit table` (no `--quantity`) emits no quantity column.
//!
//! Phase 2b of
//! docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation and a single SCALAR
/// generated quantity `peak = max(I / N)`.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  peak = max(I / N)
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(algorithm_block: &str, output_dir: &str) -> String {
    format!(
        r#"output_dir = "{output_dir}"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

{algorithm_block}
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The single fit segment dir under `<results>/fits/`.
fn fit_segment_dir(results: &Path) -> PathBuf {
    let fits = results.join("fits");
    std::fs::read_dir(&fits)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a fit segment")
}

/// Parse a CSV: header + first data row, returning the cell under `column`.
fn csv_cell(csv: &str, column: &str) -> Option<String> {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next()?.split(',').collect();
    let idx = header.iter().position(|c| *c == column)?;
    let row: Vec<&str> = lines.next()?.split(',').collect();
    row.get(idx).map(|s| s.to_string())
}

/// The `as_fitted` row's q50 from a scalar quantity TSV (columns by header name).
fn tsv_as_fitted_q50(tsv: &Path) -> f64 {
    let text = std::fs::read_to_string(tsv).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let scen_i = header.iter().position(|c| *c == "scenario").unwrap();
    let q50_i = header.iter().position(|c| *c == "q50").unwrap();
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[scen_i] == "as_fitted" {
            return cols[q50_i].parse::<f64>().unwrap();
        }
    }
    panic!("no as_fitted row in {}", tsv.display());
}

const PGAS: &str = r#"[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#;

#[test]
fn quantity_derives_on_demand_then_reads_existing() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_table_q_derive_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml(PGAS, "results")).unwrap();

    // A PGAS fit, NOT predicted.
    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "pgas fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let results = tmp.join("results");
    let seg = fit_segment_dir(&results);
    let peak_tsv = seg.join("quantities").join("peak.tsv");
    assert!(
        !peak_tsv.exists(),
        "precondition: the fit must not be predicted yet (no quantities/peak.tsv)"
    );

    // ── (a) derive-on-demand: `--quantity peak` runs predict, fills the column.
    let out = run(
        &bin,
        &tmp,
        &["fit", "table", "results/fits", "--quantity", "peak", "--format", "csv"],
    );
    assert!(
        out.status.success(),
        "fit table --quantity failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let csv = String::from_utf8_lossy(&out.stdout);
    let header = csv.lines().next().unwrap();
    assert!(
        header.split(',').any(|c| c == "peak"),
        "CSV header gained a `peak` column: {header}"
    );
    let cell = csv_cell(&csv, "peak").expect("a peak cell in the single data row");
    let peak: f64 = cell
        .parse()
        .unwrap_or_else(|_| panic!("peak cell must parse as f64, got {cell:?}"));
    assert!(
        peak.is_finite() && peak > 0.0,
        "derived peak median must be finite and positive, got {peak}"
    );
    // The derive populated the fit's `quantities/` output on disk.
    assert!(
        peak_tsv.exists(),
        "--quantity must have DERIVED quantities/peak.tsv via fit predict"
    );

    // ── (b) read-existing: a second table read returns the q50 the derive wrote.
    let out = run(
        &bin,
        &tmp,
        &["fit", "table", "results/fits", "--quantity", "peak", "--format", "json"],
    );
    assert!(out.status.success(), "fit table --format json failed");
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = doc["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 1, "one fit → one row");
    let json_peak = rows[0]["quantities"]["peak"]
        .as_f64()
        .expect("quantities.peak present and numeric in the JSON row");
    let tsv_q50 = tsv_as_fitted_q50(&peak_tsv);
    assert!(
        (json_peak - tsv_q50).abs() < 1e-9,
        "read path must return the q50 the derive wrote: json={json_peak} tsv={tsv_q50}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn quantity_if2_renders_dash_and_default_omits_column() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_table_q_if2_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();

    let if2 = r#"[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 20
cooling = 0.7
"#;
    std::fs::write(tmp.join("fit.toml"), fit_toml(if2, "results")).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "if2 fit run failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let seg = fit_segment_dir(&tmp.join("results"));

    // ── (c) IF2 → `—`: predict refuses an optimizer fit; the table still renders.
    let out = run(&bin, &tmp, &["fit", "table", "results/fits", "--quantity", "peak"]);
    assert!(
        out.status.success(),
        "fit table --quantity on an IF2 fit must not crash:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.lines().next().unwrap().contains("peak"),
        "the `peak` header column is present even for an unfillable row: {text}"
    );
    assert!(
        text.contains('—'),
        "an IF2 fit's quantity cell renders the em-dash sentinel: {text}"
    );
    // Not derivable → no spawn → no derived output on disk.
    assert!(
        !seg.join("quantities").join("peak.tsv").exists(),
        "an optimizer fit must not derive a quantities TSV"
    );

    // ── (d) default `fit table` (no --quantity) emits no `peak` column.
    let out = run(&bin, &tmp, &["fit", "table", "results/fits", "--format", "csv"]);
    assert!(out.status.success(), "default fit table failed");
    let csv = String::from_utf8_lossy(&out.stdout);
    let header = csv.lines().next().unwrap();
    assert!(
        !header.split(',').any(|c| c == "peak"),
        "the default table must not add a quantity column: {header}"
    );
    assert!(
        header.ends_with("loglik_type"),
        "default header is the pre-existing one (ends at loglik_type): {header}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
