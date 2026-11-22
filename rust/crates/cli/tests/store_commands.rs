//! `camdl status`, `verify`, `clean` and `reindex` against run stores built by
//! real `pfilter` and `simulate` runs: completed results, a run that failed, a
//! run killed while it filtered, a cell killed while it was staged, and a
//! result `--force` moved to quarantine.
//!
//! Drives the release binary; each test has a store of its own under a
//! temporary directory.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

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
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 10 'days }
"#;

/// A working directory holding the compiled model and its data, and the run
/// store `store/` under it, removed when dropped.
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
        let dir = std::env::temp_dir().join(format!("camdl_store_commands_{tag}_{}_{ns}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("sir.camdl")).output().unwrap();
        assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
        std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
        std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
        Project { dir }
    }

    fn store(&self) -> PathBuf {
        self.dir.join("store")
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut c = Command::new(camdl_bin());
        c.env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_OUTPUT_DIR")
            .env("CAMDL_STORE", self.store())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .current_dir(&self.dir)
            .args(args);
        c
    }

    fn camdl(&self, args: &[&str]) -> Output {
        self.command(args).output().expect("spawn camdl")
    }

    /// `camdl <args>`, which must succeed; its stdout.
    fn ok(&self, args: &[&str]) -> String {
        let out = self.camdl(args);
        assert!(out.status.success(), "camdl {args:?} exited {:?}:\n{}", out.status.code(), String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    }

    /// A filter run of 30 particles at step 1 under `seed`, plus `extra`.
    fn pfilter_args<'a>(seed: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
        Self::pfilter_with("30", "1", seed, extra)
    }

    fn pfilter_with<'a>(particles: &'a str, dt: &'a str, seed: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
        let mut v = vec![
            "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
            "--data", "cases.tsv", "--obs", "cases", "--particles", particles, "--dt", dt, "--seed", seed,
            "--progress", "none",
        ];
        v.extend_from_slice(extra);
        v
    }

    fn simulate_args<'a>(seed: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
        let mut v = vec![
            "simulate", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
            "--seed", seed,
        ];
        v.extend_from_slice(extra);
        v
    }

    /// Starts `camdl <args>` and returns it once `ready` holds of the store,
    /// within a minute.
    fn start_until(&self, args: &[&str], ready: impl Fn(&Path) -> bool) -> Killed {
        let mut child = Killed(self.command(args).stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn camdl"));
        let deadline = Instant::now() + Duration::from_secs(60);
        while !ready(&self.store()) {
            if let Some(status) = child.0.try_wait().unwrap() {
                panic!("camdl {args:?} ended ({status}) before the store showed it running");
            }
            assert!(Instant::now() < deadline, "camdl {args:?}: the store did not show it running within a minute");
            std::thread::sleep(Duration::from_millis(20));
        }
        child
    }
}

/// A child process, killed and reaped when dropped, so that a failing
/// assertion leaves no run going.
struct Killed(Child);

impl Killed {
    /// Kills the process and waits for it, so that its owner is dead to the
    /// store's liveness check.
    fn kill(mut self) {
        self.0.kill().unwrap();
        self.0.wait().unwrap();
    }
}

impl Drop for Killed {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// One stored result: its folder, kind, run id and recorded status.
#[derive(Debug, Clone, PartialEq)]
struct Found {
    dir: PathBuf,
    kind: String,
    run_id: String,
    status: String,
}

/// Every folder under `root` holding an `identity.json` and a `run.json`,
/// outside the store's dot-named directories.
fn results(root: &Path) -> Vec<Found> {
    fn visit(dir: &Path, out: &mut Vec<Found>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            // A claim being made writes `identity.json` before `run.json`.
            if let (Ok(id), Ok(run)) = (std::fs::read(p.join("identity.json")), std::fs::read(p.join("run.json"))) {
                let id: serde_json::Value = serde_json::from_slice(&id).unwrap();
                let run: serde_json::Value = serde_json::from_slice(&run).unwrap();
                out.push(Found {
                    dir: p.clone(),
                    kind: id["kind"].as_str().unwrap().to_string(),
                    run_id: id["run_id"].as_str().unwrap().to_string(),
                    status: run["status"].as_str().unwrap().to_string(),
                });
            }
            visit(&p, out);
        }
    }
    let mut out = Vec::new();
    visit(root, &mut out);
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    out
}

fn of_kind(root: &Path, kind: &str) -> Vec<Found> {
    results(root).into_iter().filter(|r| r.kind == kind).collect()
}

/// The entries of `dir`, by name; empty when it does not exist.
fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// The claims under the store's root `.staging/`: folders named by a run id
/// that hold a `run.json`. A claim is built under a temporary name and
/// renamed, so a folder under any other name is a claim not yet made.
fn staged(store: &Path) -> Vec<String> {
    entries(&store.join(".staging"))
        .into_iter()
        .filter(|n| n.len() == 64 && n.chars().all(|c| c.is_ascii_hexdigit()))
        .filter(|n| store.join(".staging").join(n).join("run.json").exists())
        .collect()
}

/// A completed filter run and a labelled two-cell simulation.
fn completed_runs(p: &Project) {
    p.ok(&Project::pfilter_args("1", &["--label", "pf"]));
    p.ok(&Project::simulate_args("1", &["--replicates", "2", "--label", "sm"]));
}

#[test]
fn the_commands_refuse_a_root_that_holds_no_store() {
    let p = Project::new("nostore");
    for args in [&["status"][..], &["verify"], &["clean"], &["reindex"]] {
        let out = p.camdl(args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(stderr.contains("no run store at"), "{args:?}: {stderr}");
    }
    assert!(!p.store().exists(), "no command made a store");
    let out = p.camdl(&["status", "--store", "elsewhere"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("elsewhere"), "--store names the root");
}

#[test]
fn verify_reports_the_damaged_result_and_only_that_one() {
    let p = Project::new("verify");
    completed_runs(&p);
    let store = p.store();
    assert_eq!(p.ok(&["verify"]), "every result verified\n");
    assert_eq!(p.ok(&["verify", "@sm"]), "every result verified\n");

    let cells = of_kind(&store, "sim");
    assert_eq!(cells.len(), 2, "{cells:?}");
    let filter = of_kind(&store, "pfilter").remove(0);
    let damaged = &cells[1];
    std::fs::write(damaged.dir.join("traj.tsv"), "t\tS\n0\t1\n").unwrap();

    let all = p.ok(&["verify"]);
    assert_eq!(all.lines().count(), 1, "one result is damaged: {all}");
    assert!(all.starts_with("damaged: ") && all.contains(&damaged.run_id[..12]) && all.contains("traj.tsv"), "{all}");
    let named = p.ok(&["verify", &damaged.run_id[..12]]);
    assert!(named.contains(&damaged.run_id[..12]), "{named}");
    assert_eq!(p.ok(&["verify", &cells[0].run_id[..12]]), "every result verified\n", "the other cell is whole");
    assert_eq!(p.ok(&["verify", &filter.run_id[..12]]), "every result verified\n");
    assert!(p.ok(&["verify", "@sm"]).contains(&damaged.run_id[..12]), "a label checks every cell of its invocation");

    let group = damaged.dir.parent().unwrap().to_string_lossy().into_owned();
    let out = p.camdl(&["verify", &group]);
    assert_eq!(out.status.code(), Some(1), "a folder that holds no result is refused: {}", String::from_utf8_lossy(&out.stdout));
    assert_eq!(results(&store).iter().map(|r| r.status.as_str()).collect::<Vec<_>>(), vec!["completed"; results(&store).len()], "verify writes nothing");
}

#[test]
fn reindex_rebuilds_the_index_from_the_records() {
    let p = Project::new("reindex");
    completed_runs(&p);
    let store = p.store();
    let n = results(&store).len();
    std::fs::remove_dir_all(store.join(".index")).unwrap();
    assert_eq!(p.ok(&["reindex"]), format!("indexed {n} results\n"));
    let logs = entries(&store.join(".index"));
    assert_eq!(logs.len(), 1, "one compacted log: {logs:?}");
    let text = std::fs::read_to_string(store.join(".index").join(&logs[0])).unwrap();
    for r in results(&store) {
        assert!(text.contains(&r.run_id), "the index holds {} {}", r.kind, r.run_id);
    }
    assert!(p.ok(&["list", "--kind", "pfilter"]).contains("pf"), "the store reads after reindex");
}

/// The store this test builds holds every state `clean` distinguishes:
/// completed results; a filter run recorded as failed, run twice so that it
/// has an earlier attempt; a completed filter run replaced under `--force`, so
/// that the quarantine holds the replaced copy; a filter run that is running,
/// then interrupted; and a cell whose process was killed while it was staged.
#[test]
fn clean_removes_only_quarantine_attempts_and_dead_staging() {
    let p = Project::new("clean");
    let store = p.store();
    completed_runs(&p);

    // A budget below one pass of the filter: the run is recorded as failed.
    let budget = Project::pfilter_args("2", &["--pf-max-substeps", "10"]);
    for _ in 0..2 {
        assert_eq!(p.camdl(&budget).status.code(), Some(3));
    }
    let failed = of_kind(&store, "pfilter").into_iter().find(|r| r.status == "failed").expect("a failed filter run");
    assert!(failed.dir.join(".attempts").is_dir(), "the second run kept the first as an attempt");

    p.ok(&Project::pfilter_args("1", &["--force"]));
    assert_eq!(entries(&store.join(".quarantine")).len(), 1, "--force moved the replaced run to quarantine");

    let completed_before: Vec<Found> = results(&store).into_iter().filter(|r| r.status == "completed").collect();

    // A long filter run, running while `status` and `clean` read the store.
    let long = Project::pfilter_with("10000", "0.000001", "3", &["--pf-max-substeps", "100000000000000"]);
    let running = |s: &Path| of_kind(s, "pfilter").iter().any(|r| r.status == "running");
    let child = p.start_until(&long, running);
    let live = of_kind(&store, "pfilter").into_iter().find(|r| r.status == "running").unwrap();
    let status = p.ok(&["status"]);
    assert!(status.contains(&format!("pfilter {}  running", &live.run_id[..12])), "{status}");
    assert!(status.contains(&format!("pfilter {}  failed", &failed.run_id[..12])), "{status}");
    p.ok(&["clean"]);
    assert!(live.dir.join(".lock").exists() && live.dir.join("run.json").exists(), "clean leaves a running run alone");
    child.kill();
    let status = p.ok(&["status"]);
    assert!(status.contains(&format!("pfilter {}  interrupted", &live.run_id[..12])), "{status}");

    // A cell killed while it was staged.
    let slow_cell = Project::simulate_args("9", &["--dt", "0.00000001"]);
    let child = p.start_until(&slow_cell, |s| !staged(s).is_empty());
    child.kill();
    let staging = staged(&store);
    assert_eq!(staging.len(), 1, "{staging:?}");
    assert_eq!(entries(&store.join(".staging")).iter().filter(|n| !n.ends_with(".lock")).count(), 1, "only the claim is staged");

    // The earlier `clean` ran while the quarantine and the attempt existed and
    // removed them; make both again, then clean once more with every state
    // present.
    p.ok(&Project::pfilter_args("1", &["--force"]));
    assert_eq!(p.camdl(&budget).status.code(), Some(3));
    assert!(failed.dir.join(".attempts").is_dir());
    assert_eq!(entries(&store.join(".quarantine")).len(), 1);

    let report = p.ok(&["clean"]);
    assert_eq!(report, "removed 1 staging folders, 1 quarantined folders, 1 earlier attempts, 0 empty directories\n");
    assert!(staged(&store).is_empty(), "the dead cell's staging folder is gone");
    assert!(entries(&store.join(".quarantine")).is_empty(), "the quarantine is empty");
    assert!(!failed.dir.join(".attempts").exists(), "the earlier attempts are gone");

    let after = results(&store);
    for r in &completed_before {
        assert!(after.iter().any(|a| a.run_id == r.run_id && a.status == "completed"), "clean kept completed {} {}", r.kind, r.run_id);
    }
    assert!(after.iter().any(|a| a.run_id == failed.run_id && a.status == "failed"), "clean kept the failed run's folder");
    assert!(after.iter().any(|a| a.run_id == live.run_id && a.status == "running"), "clean kept the interrupted run's folder");
    assert_eq!(p.ok(&["verify"]), "every result verified\n");
    let status = p.ok(&["status"]);
    assert!(status.contains(&format!("pfilter {}  interrupted", &live.run_id[..12])), "{status}");
    assert!(status.contains(&format!("pfilter {}  failed", &failed.run_id[..12])), "{status}");
}
