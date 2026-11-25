//! A command that only reads the run store writes nothing to it: resolving a
//! full run id goes through the index and confirms the folder it names, and a
//! prefix walks the result folders, and neither rewrites `.index/`.
//!
//! Drives the release binary on a model `camdlc` compiles first; each test has
//! a store of its own under a temporary directory.

use std::collections::BTreeMap;
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

/// A working directory holding the compiled model and the run store
/// `store/`, removed when dropped.
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
        let dir = std::env::temp_dir().join(format!("camdl_read_writes_no_index_{tag}_{}_{ns}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("sir.camdl")).output().unwrap();
        assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
        std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
        Project { dir }
    }

    fn camdl(&self, args: &[&str]) -> Output {
        let out = Command::new(camdl_bin())
            .current_dir(&self.dir)
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_LINK_ROOT")
            .env("CAMDL_STORE", self.dir.join("store"))
            .args(args)
            .output()
            .expect("spawn camdl");
        assert!(out.status.success(), "camdl {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        out
    }

    /// Simulates the model once and returns the run id of the cell it stores.
    fn simulate(&self) -> String {
        self.camdl(&["simulate", "sir.ir.json", "--param", "beta=0.5", "--param", "gamma=0.2", "--seed", "1"]);
        let record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(self.dir.join("store/.invocations/1.json")).unwrap()).unwrap();
        let sims: Vec<&str> = record["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == "sim")
            .map(|r| r["run_id"].as_str().unwrap())
            .collect();
        assert_eq!(sims.len(), 1, "{record}");
        sims[0].to_string()
    }

    /// Every file under `.index/`, by name, with its bytes.
    fn index(&self) -> BTreeMap<String, Vec<u8>> {
        std::fs::read_dir(self.dir.join("store/.index"))
            .unwrap()
            .flatten()
            .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
            .collect()
    }
}

#[test]
fn camdl_path_of_a_full_run_id_leaves_the_index_unchanged() {
    let p = Project::new("full");
    let run_id = p.simulate();
    let before = p.index();
    assert!(!before.is_empty(), "the simulation wrote its index line");
    let out = p.camdl(&["path", &run_id, "traj"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains(&run_id[..12]), "{}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(p.index(), before, "a read by run id writes nothing to .index/");
}

#[test]
fn camdl_path_of_a_prefix_leaves_the_index_unchanged() {
    let p = Project::new("prefix");
    let run_id = p.simulate();
    let before = p.index();
    let out = p.camdl(&["path", &run_id[..10], "traj"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains(&run_id[..12]), "{}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(p.index(), before, "a read by prefix writes nothing to .index/");
}

/// The index is a cache: a full run id whose index line names a folder that
/// no longer holds it is found by a walk.
#[test]
fn a_full_run_id_the_index_misplaces_is_found_by_a_walk() {
    let p = Project::new("stale");
    let run_id = p.simulate();
    for (name, bytes) in p.index() {
        let text = String::from_utf8(bytes).unwrap().replace("\"folder\":\"", "\"folder\":\"gone/");
        std::fs::write(p.dir.join("store/.index").join(name), text).unwrap();
    }
    let before = p.index();
    let out = p.camdl(&["path", &run_id, "traj"]);
    assert!(String::from_utf8_lossy(&out.stdout).contains(&run_id[..12]), "{}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(p.index(), before);
}

/// Every file under the store, by its path relative to the store, with its
/// bytes.
fn tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.strip_prefix(root).unwrap().display().to_string(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// A read whose result no closed record lists computed or reused reads the
/// running records, and that read writes nothing: no lock file is written
/// where `.invocations/` exists, and no `.invocations/` is made where there
/// is none.
#[test]
fn reading_the_running_records_writes_nothing() {
    let p = Project::new("running");
    let run_id = p.simulate();
    let store = p.dir.join("store");
    let record = store.join(".invocations/1.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    for r in v["results"].as_array_mut().unwrap() {
        r["outcome"] = "failed".into();
        r["failure"] = serde_json::json!({ "operation": null, "reason": "test" });
    }
    std::fs::write(&record, serde_json::to_vec(&v).unwrap()).unwrap();
    let before = tree(&store);
    p.camdl(&["path", &run_id, "traj"]);
    assert_eq!(tree(&store).keys().collect::<Vec<_>>(), before.keys().collect::<Vec<_>>());
    assert_eq!(tree(&store), before, "a read of the running records writes nothing to the store");

    std::fs::remove_dir_all(store.join(".invocations")).unwrap();
    let before = tree(&store);
    p.camdl(&["path", &run_id, "traj"]);
    assert!(!store.join(".invocations").exists(), "a read makes no .invocations/");
    assert_eq!(tree(&store), before);
}
