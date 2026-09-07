//! gh#833: `fit run` walks the observation axis the bound observation model
//! built — the union of every declared period's boundaries — not a list
//! rebuilt from the rows' labels.
//!
//! Under `covers = day(time)` a row labelled `D` covers `[D, D+1)` and is
//! scored at `D+1`, so the axis is the labels shifted by a day plus the first
//! period's opening boundary. A driver that rebuilt its schedule from the
//! labels and indexed the observation model by its own position would agree
//! with the model only under `closing_at`, where a row's stop is its label;
//! under `day(time)` it is one boundary short and never scores the last row,
//! and under a form with an offset it scores every row at the wrong boundary.
//! The two `day` e2e tests before this one went through `pfilter`, which steps
//! from the observation model itself; these go through the PGAS stage of
//! `fit run`, whose schedule the driver builds — the path where the two lists
//! were built separately.
//!
//! The observable: the stage's `filter_ess.tsv` has one row per observation
//! boundary the driver scored, with its time, and the saved path in
//! `chain_1/trajectories.tsv` runs to the last boundary the driver stepped to.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_coversday_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

const T_END: u32 = 60;

/// One declared form under test: the `covers` line, the labels the file
/// carries, and the boundaries the declaration puts on the axis.
struct Form {
    tag: &'static str,
    covers: &'static str,
    labels: Vec<u32>,
    /// `[start, stop)` per row, in axis units.
    periods: Vec<(u32, u32)>,
}

impl Form {
    /// Every boundary: each row's stop, plus the first row's opening.
    fn axis(&self) -> BTreeSet<u32> {
        let mut b: BTreeSet<u32> = self.periods.iter().map(|&(_, stop)| stop).collect();
        b.insert(self.periods[0].0);
        b
    }
    fn last_stop(&self) -> u32 {
        self.periods.last().unwrap().1
    }
}

/// `day(time)`: rows labelled 3..=42, each `[D, D+1)`.
fn day_form() -> Form {
    let labels: Vec<u32> = (3..=42).collect();
    Form {
        tag: "day",
        covers: "day(time)",
        periods: labels.iter().map(|&d| (d, d + 1)).collect(),
        labels,
    }
}

/// `starting_on(time, 2 'days)`: rows labelled 3, 5, …, 41, each `[D, D+2)`.
fn starting_on_form() -> Form {
    let labels: Vec<u32> = (3..=41).step_by(2).collect();
    Form {
        tag: "starting_on",
        covers: "starting_on(time, 2 'days)",
        periods: labels.iter().map(|&d| (d, d + 2)).collect(),
        labels,
    }
}

/// `ending_on(time, 2 'days)`: rows labelled 4, 6, …, 42, each `[D−1, D+1)`
/// — the labelled day included, the day before it too.
fn ending_on_form() -> Form {
    let labels: Vec<u32> = (4..=42).step_by(2).collect();
    Form {
        tag: "ending_on",
        covers: "ending_on(time, 2 'days)",
        periods: labels.iter().map(|&d| (d - 1, d + 1)).collect(),
        labels,
    }
}

/// A closed SIR whose incidence stream carries the form's declaration, and a
/// counts file with the form's labels.
fn write_fixture(dir: &Path, form: &Form) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = incidence(infection)
    covers        = {covers}
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = {T_END} 'days }}
"#, covers = form.covers);
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let mut data = String::from("time\tcases\n");
    for &d in &form.labels {
        let x = f64::from(d);
        let n = (40.0 * (-((x - 20.0) / 8.0).powi(2)).exp()).round() as i64 + 2;
        data.push_str(&format!("{d}\t{n}\n"));
    }
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path, data).unwrap();
    (ir_path, data_path)
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path) -> (PathBuf, PathBuf) {
    let out = dir.join("results");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 30
sweeps = 6
burn_in = 2
thin = 1
"#,
        out = out.display(), ir = ir.display(), data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    (p, out)
}

fn stage_leaf(out: &Path) -> PathBuf {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return d;
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

/// A TSV as header + rows, skipping `#` provenance lines.
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("header").split('\t').map(String::from).collect();
    let rows = lines.map(|l| l.split('\t').map(String::from).collect::<Vec<_>>()).collect();
    (header, rows)
}

fn col<'a>(header: &[String], row: &'a [String], name: &str) -> &'a str {
    let i = header.iter().position(|c| c == name)
        .unwrap_or_else(|| panic!("no `{name}` column; header: {header:?}"));
    &row[i]
}

/// Run the PGAS stage on one form and check the driver walked the declared
/// boundaries: one filter-ESS row per boundary, and a path that reaches the
/// last row's stop.
fn assert_pgas_walks_the_declared_boundaries(form: &Form) {
    let bin = camdl_bin();
    let tmp = tempdir(form.tag);
    let (ir, data) = write_fixture(tmp.path(), form);
    let (fit, out) = write_fit_toml(tmp.path(), &ir, &data);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit).arg("--seed").arg("7")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().expect("spawn camdl");
    assert!(r.status.success(), "[{}] fit run failed: {}", form.tag, String::from_utf8_lossy(&r.stderr));
    let stage = stage_leaf(&out);

    let (header, rows) = read_tsv(&stage.join("filter_ess.tsv"));
    let scored: BTreeSet<u32> = rows.iter()
        .filter(|r| col(&header, r, "chain") == "1")
        .map(|r| col(&header, r, "time").parse::<f64>().expect("time parses").round() as u32)
        .collect();
    let labels: BTreeSet<u32> = form.labels.iter().copied().collect();
    assert_eq!(
        scored, form.axis(),
        "[{}] the driver's observation axis must be the declared periods' boundaries; \
         a driver rebuilt from the labels would walk {labels:?}", form.tag
    );

    let (h, paths) = read_tsv(&stage.join("chain_1/trajectories.tsv"));
    let last_t = paths.iter()
        .map(|r| col(&h, r, "time").parse::<f64>().expect("time parses"))
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(last_t, f64::from(form.last_stop()),
        "[{}] the path must run to the last row's stop, {}; it ran to {last_t}", form.tag, form.last_stop());
}

#[test]
fn pgas_fit_run_scores_every_day_row_at_the_boundary_its_period_closes() {
    if camdlc_bin().is_none() { return }
    assert_pgas_walks_the_declared_boundaries(&day_form());
}

/// The two named forms have run through the loader's unit tests but, until
/// here, never through a fitting driver.
#[test]
fn pgas_fit_run_walks_starting_on_and_ending_on_periods() {
    if camdlc_bin().is_none() { return }
    assert_pgas_walks_the_declared_boundaries(&starting_on_form());
    assert_pgas_walks_the_declared_boundaries(&ending_on_form());
}
