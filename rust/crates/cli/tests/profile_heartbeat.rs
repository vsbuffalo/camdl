//! `camdl profile` writes a run-level liveness heartbeat (`progress.json`,
//! gh#278) at the profile **base** directory, so `camdl-watcher` can monitor a
//! profile the way it monitors a fit stage: a background thread refreshes the
//! file every few seconds with `state = running{phase: profiling, step, total}`
//! (the `updated_at` freshness is the liveness signal), and a clean finish
//! writes the terminal `state = done`.
//!
//! This pins the wiring: a completed profile leaves exactly one `progress.json`
//! at the base (alongside the per-cell point dirs and `fit.meta.json`, not
//! inside a cell), and it parses to the terminal `done` state. The intermediate
//! `running{profiling, step, total}` shape is covered by the unit tests in
//! `io::progress`; catching it here would be timing-flaky, so this asserts the
//! durable end state and the base-level placement.

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

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_profile_hb_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin();
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
scenarios {
  baseline { set = { beta = 0.3  gamma = 0.1  N0 = 1000 } }
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

#[test]
fn profile_writes_liveness_heartbeat_at_base() {
    let bin = camdl_bin();
    let tmp = tempdir("hb");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("out");

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", &root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--sweep", "beta=lin(0.2,0.4,2)",
            "--sweep", "gamma=lin(0.05,0.15,2)",
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "5",
            "--starts", "1",
            "--rw-sd", "auto",
            "--seed", "1",
        ])
        .output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile must exit 0.\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));

    // Exactly one progress.json, and it sits at the base (its dir also holds
    // fit.meta.json and the per-cell point dirs) — not inside a cell leaf.
    let mut found: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|n| n.to_str()) == Some("progress.json") {
                    found.push(p);
                }
            }
        }
    }
    assert_eq!(found.len(), 1,
        "expected exactly one run-level progress.json, found {}: {:?}", found.len(), found);
    let pj = &found[0];
    let base_dir = pj.parent().unwrap();
    assert!(base_dir.join("fit.meta.json").is_file(),
        "progress.json must sit at the base dir (next to fit.meta.json), got {}", pj.display());
    // The base dir also roots the per-cell leaves.
    let has_point_dirs = std::fs::read_dir(base_dir).unwrap().flatten()
        .any(|e| e.path().is_dir());
    assert!(has_point_dirs, "base dir must also contain the per-cell point dirs");

    // Parses, and a completed run ends in the terminal `done` state.
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(pj).expect("read progress.json"))
        .expect("progress.json must be valid JSON");
    assert_eq!(v.get("state").and_then(|s| s.as_str()), Some("done"),
        "a completed profile's heartbeat must be terminal `done`, got: {}", v);
    assert!(v.get("updated_at").and_then(|u| u.as_u64()).is_some(),
        "heartbeat must carry updated_at (the watcher's liveness signal): {}", v);
}
