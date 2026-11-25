//! Every command that stores a result applies one rule to `--label`: the
//! store parses it when the command's store request is made. A label that
//! breaks the rule is refused before anything is stored, so no invocation
//! records a label that no `@label` handle reaches.
//!
//! Drives the release binary on a model `camdlc` compiles first; each
//! test has a store of its own under a temporary directory.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 5 'days }
"#;

/// A working directory holding the compiled model, removed when dropped.
struct Project {
    dir: PathBuf,
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Project {
    fn new(tag: &str) -> Self {
        let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("camdl_store_parses_label_{tag}_{}_{ns}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("sir.camdl")).output().unwrap();
        assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
        std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
        Project { dir }
    }

    fn simulate(&self, label: &str) -> Output {
        Command::new(camdl_bin())
            .current_dir(&self.dir)
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_LINK_ROOT")
            .env("CAMDL_STORE", self.dir.join("store"))
            .args(["simulate", "sir.ir.json", "--param", "beta=0.5", "--param", "gamma=0.2", "--seed", "1"])
            .args(["--output", "traj.tsv", "--label", label])
            .output()
            .expect("spawn camdl")
    }

    /// The invocation records the store holds.
    fn records(&self) -> Vec<serde_json::Value> {
        let dir = self.dir.join("store").join(".invocations");
        let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
        entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json") && !p.to_string_lossy().ends_with(".progress.json"))
            .map(|p| serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap())
            .collect()
    }
}

#[test]
fn simulate_refuses_a_label_no_handle_reaches() {
    for bad in ["f~1", "..", "a/b"] {
        let p = Project::new("bad");
        let out = p.simulate(bad);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "`simulate --label {bad}` must be refused; stderr: {stderr}");
        assert!(stderr.contains("label"), "the refusal names the label: {stderr}");
        assert!(p.records().is_empty(), "a refused label records no invocation: {:?}", p.records());
    }
}

#[test]
fn simulate_records_a_label_the_rule_accepts_trimmed() {
    let p = Project::new("good");
    let out = p.simulate("  take 1, a-b_c.d  ");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let records = p.records();
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0]["label"], "take 1, a-b_c.d");
}
