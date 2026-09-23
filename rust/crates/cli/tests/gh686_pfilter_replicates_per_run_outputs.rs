//! gh#686: `camdl pfilter --replicates N` (N > 1) returned before the writers
//! of every per-run output, so `--pf-health`, `--trace` and the other
//! single-run files were silently not written, at exit 0. A stale file from an
//! earlier run then reads as this run's. The combination is now refused.
//!
//! End-to-end via the release `camdl` binary; skipped when it or `camdlc.exe`
//! is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gh686_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// A SIR model, its compiled IR, parameters and a short prevalence series.
fn fixture(dir: &Path) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let camdlc = camdlc()?;
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;
    let model = dir.join("sir.camdl");
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("sir.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let params = dir.join("theta.toml");
    std::fs::write(&params, "beta = 0.6\ngamma = 0.2\nN0 = 1000\n").unwrap();
    let data = dir.join("cases.tsv");
    std::fs::write(&data, "time\tcases\n2\t14\n4\t19\n6\t25\n8\t31\n10\t37\n").unwrap();
    Some((ir, params, data))
}

fn pfilter(dir: &Path, fx: &(PathBuf, PathBuf, PathBuf), extra: &[&str]) -> std::process::Output {
    let (ir, params, data) = fx;
    Command::new(camdl_bin())
        .arg("pfilter").arg(ir)
        .arg("--params").arg(params)
        .arg("--data").arg(data)
        .args(["--particles", "50", "--seed", "5"])
        .args(extra)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap()
}

#[test]
fn replicates_with_a_per_run_output_is_refused() {
    let tmp = tempdir("refuse");
    if !camdl_bin().exists() { return; }
    let Some(fx) = fixture(tmp.path()) else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    // Every per-run output the replicate path used to drop.
    for (flag, value) in [
        ("--pf-health", "health.tsv"),
        ("--trace", "trace.tsv"),
        ("--save-final-state", "final.tsv"),
        ("--save-paths", "paths.tsv"),
        ("--save-filtering", "filtering.tsv"),
        ("--save-prequential", "preq"),
    ] {
        let out = pfilter(tmp.path(), &fx, &["--replicates", "2", flag, value]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(),
            "`--replicates 2 {flag}` must be refused, not exit 0 without the file; \
             stderr:\n{stderr}");
        assert!(stderr.contains(flag),
            "the refusal must name the flag it refuses ({flag}); stderr:\n{stderr}");
        assert!(!tmp.path().join(value).exists(), "{flag}: no file may be written");
    }
}

/// Negative control: a single run writes the health report, so the refusal is
/// about `--replicates`, not the flag.
#[test]
fn a_single_run_writes_the_health_report() {
    let tmp = tempdir("single");
    if !camdl_bin().exists() { return; }
    let Some(fx) = fixture(tmp.path()) else { return };
    let out = pfilter(tmp.path(), &fx, &["--replicates", "1", "--pf-health", "health.tsv"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(tmp.path().join("health.tsv").exists());
}
