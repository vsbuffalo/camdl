//! An invocation record of another format, such as one an earlier build
//! wrote, is not read, and the command says so once. Its results stay
//! reusable, since reuse goes by identity; its label and its listings are not
//! seen. A closed record that lists a result as still planned is refused the
//! same way.
//!
//! Drives the release binary on a model `camdlc` compiles first; each test
//! has a store of its own under a temporary directory.

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
        let dir = std::env::temp_dir().join(format!("camdl_invocation_record_format_{tag}_{}_{ns}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("sir.camdl")).output().unwrap();
        assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
        std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
        Project { dir }
    }

    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(camdl_bin())
            .current_dir(&self.dir)
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_LINK_ROOT")
            .env("CAMDL_STORE", self.dir.join("store"))
            .args(args)
            .output()
            .expect("spawn camdl")
    }

    fn simulate(&self, label: &str) -> Output {
        self.camdl(&[
            "simulate", "sir.ir.json", "--param", "beta=0.5", "--param", "gamma=0.2", "--seed", "1", "--output", "traj.tsv",
            "--label", label,
        ])
    }

    fn record_path(&self, seq: u64) -> PathBuf {
        self.dir.join("store").join(".invocations").join(format!("{seq}.json"))
    }

    fn edit_record(&self, seq: u64, edit: impl FnOnce(&mut serde_json::Value)) {
        let path = self.record_path(seq);
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        edit(&mut v);
        std::fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    }
}

fn ok(out: &Output) -> String {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A record of format 1, as an earlier build wrote it, is not read: its label
/// names nothing, and the command that looked says so once, naming the file
/// and both formats.
#[test]
fn a_record_of_another_format_is_reported_once_and_not_read() {
    let p = Project::new("format");
    ok(&p.simulate("a"));
    p.edit_record(1, |v| v["format"] = 1.into());
    let out = p.camdl(&["path", "@a", "traj"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "`@a` names a record this build does not read; stdout: {}", String::from_utf8_lossy(&out.stdout));
    let record = p.record_path(1).display().to_string();
    let said: Vec<&str> = stderr.lines().filter(|l| l.contains("not read")).collect();
    assert_eq!(said.len(), 1, "the unread record is reported once: {stderr}");
    assert!(stderr.contains(&record) && stderr.contains("format 1") && stderr.contains("format 2"), "{stderr}");
}

/// The results an unread record lists stay reusable: the same simulation
/// again reuses its stored result, and its own record is read.
#[test]
fn a_result_an_unread_record_lists_is_reused() {
    let p = Project::new("reuse");
    ok(&p.simulate("a"));
    p.edit_record(1, |v| v["format"] = 1.into());
    let out = p.simulate("b");
    ok(&out);
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(p.record_path(2)).unwrap()).unwrap();
    let outcomes: Vec<&str> = record["results"].as_array().unwrap().iter().map(|r| r["outcome"].as_str().unwrap()).collect();
    assert!(outcomes.iter().all(|o| *o == "reused"), "{record:#}");
    ok(&p.camdl(&["path", "@b", "traj"]));
}

/// A closed record that lists a result as planned is not a record of this
/// format: it is reported and not read.
#[test]
fn a_closed_record_with_a_planned_result_is_refused() {
    let p = Project::new("planned");
    ok(&p.simulate("a"));
    p.edit_record(1, |v| v["results"][0]["outcome"] = "planned".into());
    let out = p.camdl(&["path", "@a", "traj"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the record is not read, so `@a` names nothing");
    assert!(stderr.contains(&p.record_path(1).display().to_string()) && stderr.contains("planned"), "{stderr}");
}

/// The record a command writes has format 2, its results' `attributes`, and
/// no `counts` or `progress`.
#[test]
fn the_record_is_format_2_with_attributes() {
    let p = Project::new("shape");
    ok(&p.simulate("a"));
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(p.record_path(1)).unwrap()).unwrap();
    assert_eq!(record["format"], 2, "{record:#}");
    assert_eq!(record["status"], "ended");
    assert!(record.get("counts").is_none() && record.get("progress").is_none(), "{record:#}");
    let results = record["results"].as_array().unwrap();
    assert!(results.iter().all(|r| r.get("attributes").is_some() && r.get("labels").is_none()), "{record:#}");
    assert!(results.iter().all(|r| r.get("failure").is_none()), "a result that did not fail carries no failure: {record:#}");
}
