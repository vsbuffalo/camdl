//! A `camdl profile` invocation leaves a record of what it will run and a
//! heartbeat under `profiles/<base>/invocations/<invocation_id>/`, outside
//! every leaf and every hash.
//!
//! Pinned from outside, on a 2×2 grid:
//!
//! - `invocation.json` lists every grid cell with its exact focal values and
//!   the `<point>` directory its leaves were actually written under;
//! - `progress.json` ends `done` (its running → done transitions and the
//!   cached-job count are pinned in `profile_invocation`'s unit tests, where
//!   the writes are deterministic);
//! - a rerun whose every job is cached writes its own record;
//! - the extra directory is not mistaken for a grid point: `camdl list` still
//!   shows one profile of the right shape, `camdl dev reindex` finds every
//!   leaf, and the collected table is unchanged;
//! - nothing in a leaf mentions the invocation, and a rerun's leaves keep
//!   their `run_id`s.

use std::collections::BTreeSet;
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
        "camdl_profile_invocation_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
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
    projected  = I
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
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&model_path)
        .output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path, "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    std::fs::write(dir.join("start.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 1000\n").unwrap();
    (ir_path, data_path)
}

const BETA: &str = "beta=0.1,0.5";
const GAMMA: &str = "gamma=0.03,0.25";

/// A 2×2 IF2 profile, two starts per cell, into `root`. Returns stdout (the
/// collected table).
fn run_profile(ir: &Path, data: &Path, root: &Path) -> String {
    let out = profile_cmd(ir, data, root).output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile must exit 0.\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn profile_cmd(ir: &Path, data: &Path, root: &Path) -> Command {
    let mut cmd = Command::new(camdl_bin());
    cmd.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--starts",
            &format!("from_params={}", ir.with_file_name("start.toml").to_string_lossy()),
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--sweep", BETA,
            "--sweep", GAMMA,
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "3",
            "--n-starts", "2",
            "--rw-sd", "auto",
            "--seed", "1",
            "--label", "inv-test",
        ]);
    cmd
}

/// An invocation camdl stops on an error records `failed` with the sentence
/// it printed, as a fit method's heartbeat does — here the collected table's
/// `--output` cannot be written, after every job has run.
#[test]
fn a_failed_invocation_records_why() {
    let tmp = tempdir("failed");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("out");
    let unwritable = tmp.path().join("no-such-dir").join("table.tsv");
    let out = profile_cmd(&ir, &data, &root)
        .args(["--output", &unwritable.to_string_lossy()])
        .output().unwrap();
    assert!(!out.status.success(), "an unwritable --output fails the command");
    let invs = invocations(&base(&root));
    assert_eq!(invs.len(), 1);
    let progress = read_json(&invs[0].join("progress.json"));
    let reason = progress["state"]["failed"]["reason"].as_str()
        .unwrap_or_else(|| panic!("a failed state: {progress:#}"));
    assert!(reason.contains("cannot write --output"), "{reason}");
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The one profile base under `root/profiles/`.
fn base(root: &Path) -> PathBuf {
    let bases: Vec<PathBuf> = std::fs::read_dir(root.join("profiles")).unwrap().flatten()
        .map(|e| e.path()).filter(|p| p.is_dir()).collect();
    assert_eq!(bases.len(), 1, "one base: {bases:?}");
    bases[0].clone()
}

/// The invocation directories under `base`, oldest first.
fn invocations(base: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(base.join("invocations")).unwrap().flatten()
        .map(|e| e.path()).collect();
    v.sort();
    v
}

/// Every `run.json` under `dir`, with its `run_id`, sorted by path.
fn leaves(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let id = read_json(&p)["run_id"].as_str().unwrap().to_string();
                out.push((p, id));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn a_profile_invocation_records_its_grid_and_ends_done() {
    let tmp = tempdir("record");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("out");
    let table_first = run_profile(&ir, &data, &root);
    let base = base(&root);

    let invs = invocations(&base);
    assert_eq!(invs.len(), 1, "one invocation → one record: {invs:?}");
    let inv = &invs[0];
    let rec = read_json(&inv.join("invocation.json"));

    assert_eq!(rec["schema"], "camdl.profile-invocation/v1");
    assert_eq!(rec["invocation_id"].as_str().unwrap(),
        inv.file_name().unwrap().to_str().unwrap(), "the id names its directory");
    assert!(rec["started_at"].as_u64().unwrap() > 1_700_000_000);
    assert_eq!(rec["label"], "inv-test");
    assert_eq!(rec["algorithm"], "if2");
    assert_eq!(rec["focal"], serde_json::json!([
        {"name": "beta", "values": [0.1, 0.5]},
        {"name": "gamma", "values": [0.03, 0.25]},
    ]));
    assert_eq!(rec["n_starts"], 2);
    assert_eq!(rec["seeds"], serde_json::json!([1]));
    assert_eq!(rec["total_jobs"], 8);

    // Every cell, with its exact focal values in --sweep order.
    let points = rec["points"].as_array().unwrap();
    let coords: Vec<serde_json::Value> = points.iter().map(|p| p["coords"].clone()).collect();
    for b in [0.1, 0.5] {
        for g in [0.03, 0.25] {
            assert!(coords.contains(&serde_json::json!({"beta": b, "gamma": g})),
                "cell beta={b} gamma={g} missing: {coords:?}");
        }
    }
    assert_eq!(points.len(), 4);
    let text = std::fs::read_to_string(inv.join("invocation.json")).unwrap();
    assert!(text.find("\"beta\"").unwrap() < text.find("\"gamma\"").unwrap());

    // Each `dir` is the point directory its leaves were written under: every
    // one exists under the base with `n_starts × seeds` leaves, and the
    // base's point directories are exactly these.
    let named: BTreeSet<String> =
        points.iter().map(|p| p["dir"].as_str().unwrap().to_string()).collect();
    assert_eq!(named.len(), 4, "four distinct point dirs: {named:?}");
    let method_dir = rec["method_dir"].as_str().unwrap();
    assert!(method_dir.starts_with("if2-"), "{method_dir}");
    for p in points {
        let dir = base.join(p["dir"].as_str().unwrap()).join(method_dir);
        let ls = leaves(&dir);
        assert_eq!(ls.len(), 2, "{} holds two starts' leaves", dir.display());
        for (run_json, _) in &ls {
            let r = read_json(run_json);
            let level = r["levels"].as_array().unwrap().iter()
                .find(|l| l["name"] == "point").unwrap().clone();
            let label = level["label"].as_str().unwrap();
            let want = format!("beta={:.4}__gamma={:.4}",
                p["coords"]["beta"].as_f64().unwrap(), p["coords"]["gamma"].as_f64().unwrap());
            assert_eq!(label, want, "{} is the leaf of the cell it is listed for",
                run_json.display());
        }
    }
    let point_dirs: BTreeSet<String> = std::fs::read_dir(&base).unwrap().flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "invocations")
        .collect();
    assert_eq!(point_dirs, named);

    // The heartbeat ended done, in the format fit runs write.
    let progress = read_json(&inv.join("progress.json"));
    assert_eq!(progress["state"], "done", "{progress:#}");
    assert!(progress["pid"].as_u64().is_some() && progress["updated_at"].as_u64().is_some());

    // Nothing in a leaf names the invocation or its files.
    let first_leaves = leaves(&base);
    assert_eq!(first_leaves.len(), 8);
    let id = rec["invocation_id"].as_str().unwrap();
    for (run_json, _) in &first_leaves {
        let s = std::fs::read_to_string(run_json).unwrap();
        for needle in [id, "invocations/", "invocation.json"] {
            assert!(!s.contains(needle), "{} names {needle}", run_json.display());
        }
    }

    // A rerun whose every job is cached writes its own record, and leaves
    // every leaf and the table as they were.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let table_again = run_profile(&ir, &data, &root);
    let invs = invocations(&base);
    assert_eq!(invs.len(), 2, "the resumed invocation has its own record: {invs:?}");
    let resumed = &invs[1];
    let rec2 = read_json(&resumed.join("invocation.json"));
    assert_eq!(rec2["total_jobs"], 8);
    assert_eq!(rec2["points"], rec["points"], "the same grid lands in the same dirs");
    assert_eq!(read_json(&resumed.join("progress.json"))["state"], "done");
    assert_eq!(leaves(&base), first_leaves, "a rerun keeps every leaf's run_id");
    assert_eq!(table_again, table_first, "the collected table reads the same leaves");

    // Readers that walk the tree do not take `invocations/` for a cell.
    let list = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .args(["list", &root.to_string_lossy(), "--kind", "profile", "--format", "json"])
        .output().unwrap();
    assert!(list.status.success(), "camdl list: {}", String::from_utf8_lossy(&list.stderr));
    // One JSON object per listed profile, one per line.
    let listed = String::from_utf8_lossy(&list.stdout);
    let rows: Vec<serde_json::Value> = listed.lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v.get("shape").is_some())
        .collect();
    assert_eq!(rows.len(), 1, "one profile listed: {listed}");
    assert_eq!(rows[0]["shape"], "2×2 starts=2", "the listed shape counts four cells");
    assert_eq!(rows[0]["focal"], "beta,gamma");
    assert_eq!(rows[0]["label"], "inv-test");
    assert!(!String::from_utf8_lossy(&list.stderr).contains("warning"),
        "no warning about the invocations dir: {}", String::from_utf8_lossy(&list.stderr));

    let reindex = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .args(["dev", "reindex", &root.to_string_lossy()])
        .output().unwrap();
    assert!(reindex.status.success());
    let said = String::from_utf8_lossy(&reindex.stdout).to_string()
        + &String::from_utf8_lossy(&reindex.stderr);
    assert!(said.contains("reindexed 8 run(s)"), "reindex finds every leaf: {said}");
}
