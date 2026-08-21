//! gh#688: the PGAS trace carries renewal resolved in time.
//!
//! `chain_<n>/trace.tsv` gains `renewal_b0 … renewal_b9` — `trajectory_renewal`
//! split into tenths of the substep series. Averaged down each column over
//! post-burn-in sweeps they are the update-rate-against-t plot that Lindsten,
//! Jordan & Schön (2014, *JMLR* 15:2145-2184, Fig. 1) and Chopin & Singh (2015,
//! *Bernoulli* 21:1855-1883) recommend in place of a rule for choosing the
//! particle count, of which there is none.
//!
//! What this pins that the in-crate tests cannot: the columns are LABELLED and
//! ALIGNED on disk. A trace whose header and value rows disagree in width, or
//! whose values are written in a different order from the header, mislabels
//! every diagnostic on the row — silently, since every column is numeric.
//!
//! The fixture runs 6 days at dt = 1, i.e. 6 substeps over 10 bins, so four
//! bins necessarily hold no substep and must render `NA` rather than `0.0` —
//! "no substep fell here" and "no substep here was renewed" are different
//! diagnoses.
//!
//! Skipped when the release binary or camdlc isn't present.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
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
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_pgas_renewal_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A tiny SIR + Poisson-obs model, 6 days at dt = 1, so PGAS runs end-to-end in
/// seconds and the substep series is shorter than the bin count.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, sweeps: usize) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0],  prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
[stages.post]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 30
sweeps = {sweeps}
burn_in = 2
"#,
        out = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// The `chain_1/trace.tsv` of the `post` stage leaf under `<out>/fits/`.
fn trace_of_post_leaf(out: &Path) -> String {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return std::fs::read_to_string(d.join("chain_1/trace.tsv"))
                        .expect("read chain_1/trace.tsv");
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

#[test]
fn pgas_trace_carries_renewal_resolved_in_time() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("bins");
    let (ir, data) = write_fixture(tmp.path());
    let out = tmp.path().join("results");

    let fit = write_fit_toml(tmp.path(), &ir, &data, 8);
    let r = Command::new(&bin)
        .arg("fit").arg("run").arg(&fit).arg("--seed").arg("1")
        .output().expect("spawn");
    assert!(r.status.success(), "PGAS run failed: {}", String::from_utf8_lossy(&r.stderr));

    let text = trace_of_post_leaf(&out);
    let mut lines = text.lines();
    let header = lines.next().expect("trace has a header line");
    let cols: Vec<&str> = header.split('\t').collect();

    // The renewal profile and the ancestor-sampling counters are on the same
    // row on purpose: the profile says WHERE the reference path is stuck, the
    // acceptance rate says WHY.
    for c in ["trajectory_renewal", "as_accept", "as_proposed"] {
        assert!(cols.contains(&c), "PGAS trace must keep the `{c}` column; header was: {header}");
    }
    let bin_cols: Vec<String> = (0..10).map(|b| format!("renewal_b{b}")).collect();
    for c in &bin_cols {
        assert!(
            cols.contains(&c.as_str()),
            "PGAS trace must carry the `{c}` time-resolved renewal column (gh#688); \
             header was: {header}"
        );
    }
    let idx = |name: &str| cols.iter().position(|&c| c == name).unwrap();
    let i_renewal = idx("trajectory_renewal");
    let bin_idx: Vec<usize> = bin_cols.iter().map(|c| idx(c)).collect();

    let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    assert!(!rows.is_empty(), "trace has no data rows");

    for row in &rows {
        let f: Vec<&str> = row.split('\t').collect();
        assert_eq!(
            f.len(), cols.len(),
            "row has {} fields but the header names {} columns — every diagnostic after the \
             mismatch is mislabelled; row was: {row}",
            f.len(), cols.len()
        );

        let renewal: f64 = f[i_renewal].parse().expect("trajectory_renewal parses");
        // 6 substeps over 10 bins: exactly the bins holding a substep report a
        // number, and each holds exactly one, so their mean IS the aggregate.
        let mut present = Vec::new();
        for (b, &i) in bin_idx.iter().enumerate() {
            if f[i] == "NA" { continue; }
            let r: f64 = f[i].parse().unwrap_or_else(|e| {
                panic!("renewal_b{b} = {:?} is neither `NA` nor a number ({e}); row was: {row}", f[i])
            });
            assert!(
                (0.0..=1.0).contains(&r),
                "renewal_b{b} is a fraction of substeps and must lie in [0,1]; got {r}"
            );
            present.push(r);
        }
        assert_eq!(
            present.len(), 6,
            "6 substeps fall in 6 of the 10 bins; the other 4 must read `NA`, not 0.0 — \
             row was: {row}"
        );
        let mean = present.iter().sum::<f64>() / present.len() as f64;
        assert!(
            (mean - renewal).abs() < 1e-4,
            "the per-bin profile must agree with the aggregate it resolves: bins mean to \
             {mean:.4}, trajectory_renewal reads {renewal:.4}; row was: {row}"
        );
    }
}
