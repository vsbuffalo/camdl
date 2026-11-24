//! `--manifest PATH`: the output manifest of `simulate`, `pfilter`, `fit run`,
//! `predict`, `score` and `profile` (run spec §2.10), read as a workflow tool
//! reads it.
//!
//! Drives the release binary on a three-compartment SIR model; each test has
//! a working directory and run store of its own.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use sha2::{Digest, Sha256};

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

const FIT: &str = r#"
[model]
camdl = "sir.ir.json"

[data.observations]
cases = "cases.tsv"

[estimate.beta]
bounds = [0.05, 2.0]
start  = 0.3
prior  = { log_normal = { mu = -1.2, sigma = 0.5 } }

[fixed]
gamma = 0.1
N0    = 1000

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 20
sweeps = 4
burn_in = 1
thin = 1

[config]
dt = 1.0
"#;

/// A working directory holding the compiled model, its data and a fit
/// config, with the run store `store/` under it.
struct Project {
    tmp: tempfile::TempDir,
}

impl Project {
    fn new() -> Project {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("sir.camdl"), MODEL).unwrap();
        let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(dir.join("sir.camdl")).output().unwrap();
        assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
        std::fs::write(dir.join("sir.ir.json"), &out.stdout).unwrap();
        std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
        std::fs::write(dir.join("fit.toml"), FIT).unwrap();
        std::fs::create_dir(dir.join("build")).unwrap();
        Project { tmp }
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir().join(rel)
    }

    fn run_in(&self, store: &str, args: &[&str]) -> Output {
        Command::new(camdl_bin())
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_STORE", self.path(store))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc_bin())
            .current_dir(self.dir())
            .args(args)
            .output()
            .expect("spawn camdl")
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_in("store", args)
    }

    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(out.status.success(), "camdl {args:?} exited {:?}:\n{}", out.status.code(), String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    }

    /// The manifest at `rel`, parsed.
    fn manifest(&self, rel: &str) -> serde_json::Value {
        let bytes = std::fs::read(self.path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        serde_json::from_slice(&bytes).unwrap()
    }
}

/// `camdl simulate` of the SIR model at beta = 0.3, plus `extra`.
fn simulate<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["simulate", "sir.ir.json", "--param", "beta=0.3"];
    v.extend_from_slice(&["--param", "gamma=0.1", "--param", "N0=1000"]);
    v.extend_from_slice(extra);
    v
}

/// `camdl simulate` of the SIR model with beta left to a sweep.
fn sweep<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["simulate", "sir.ir.json", "--param", "gamma=0.1", "--param", "N0=1000"];
    v.extend_from_slice(extra);
    v
}

fn results(m: &serde_json::Value) -> &Vec<serde_json::Value> {
    m["results"].as_array().expect("results is an array")
}

fn kinds(m: &serde_json::Value) -> Vec<&str> {
    results(m).iter().map(|r| r["kind"].as_str().unwrap()).collect()
}

/// Every result's `run.json`, by run id, as the store holds it.
fn run_records(store: &Path) -> std::collections::BTreeMap<String, serde_json::Value> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "identity.json") {
                let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                let run: serde_json::Value = serde_json::from_slice(&std::fs::read(p.with_file_name("run.json")).unwrap()).unwrap();
                out.insert(identity["run_id"].as_str().unwrap().to_string(), run);
            }
        }
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// The manifest names every result of the invocation and each result file
/// with the digest and size its `run.json` records, and no provenance file.
#[test]
fn manifest_digests_are_the_store_digests() {
    let p = Project::new();
    p.ok(&simulate(&["--seed", "1", "--obs-dir", "obs", "--label", "one", "--manifest", "build/one.json"]));
    let m = p.manifest("build/one.json");
    assert_eq!(m["schema"], "camdl.output-manifest/v1");
    assert_eq!(m["label"], "one");
    assert_eq!(m["camdl"]["engine_id"].as_str().unwrap().len(), 32);
    assert_eq!(m["camdl"]["ambient_key"].as_str().unwrap().len(), 64);
    assert_eq!(kinds(&m), ["compiled_model", "sim", "obs"]);
    let records = run_records(&p.path("store"));
    for r in results(&m) {
        let id = r["run_id"].as_str().unwrap();
        assert_eq!(id.len(), 64, "a run id is 64 hex digits");
        assert_eq!(r["contract"], 1);
        let run = &records[id];
        let result_files: Vec<&str> = run["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, f)| f["role"] == "result")
            .map(|(n, _)| n.as_str())
            .collect();
        let listed: Vec<&str> = r["files"].as_object().unwrap().values().map(|f| f["file"].as_str().unwrap()).collect();
        let mut listed_sorted = listed.clone();
        listed_sorted.sort();
        let mut want = result_files.clone();
        want.sort();
        assert_eq!(listed_sorted, want, "{} {id}: the manifest lists exactly the result files", r["kind"]);
        for f in r["files"].as_object().unwrap().values() {
            let stored = &run["files"][f["file"].as_str().unwrap()];
            assert_eq!(f["sha256"], stored["sha256"], "the digest is run.json's");
            assert_eq!(f["bytes"], stored["bytes"], "the size is run.json's");
        }
    }
    let sim = &results(&m)[1];
    assert_eq!(sim["files"]["traj"]["file"], "traj.tsv", "a cell's trajectory is named `traj`");
    assert!(sim["files"].get("provenance").is_none(), "a provenance file is not listed");
    let obs = &results(&m)[2];
    assert_eq!(obs["upstream"], sim["run_id"], "a by-product names its upstream");
    assert_eq!(obs["files"]["cases"]["file"], "cases.tsv", "an observation stream is named by its stream");
    assert!(results(&m)[0]["files"]["model"]["file"] == "model.ir.json", "the compiled model's file is `model`");
}

/// A workflow step reads a run id and a stored name from the manifest, reads
/// the file through camdl, and gets the bytes whose digest the manifest
/// records.
#[test]
fn a_workflow_reads_a_file_named_by_the_manifest() {
    let p = Project::new();
    p.ok(&simulate(&["--seed", "3", "--manifest", "build/sim.json"]));
    let m = p.manifest("build/sim.json");
    let sim = results(&m).iter().find(|r| r["kind"] == "sim").expect("a sim entry");
    let id = sim["run_id"].as_str().unwrap();
    let traj = &sim["files"]["traj"];
    let out = p.run(&["cat", id, traj["file"].as_str().unwrap()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(sha256_hex(&out.stdout), traj["sha256"].as_str().unwrap());
    assert_eq!(out.stdout.len() as u64, traj["bytes"].as_u64().unwrap());
}

/// A cache hit writes the manifest again, with the bytes the computation
/// wrote; so does the same command against another store root.
#[test]
fn manifest_bytes_are_identical_on_a_cache_hit_and_for_two_store_roots() {
    let p = Project::new();
    let args = simulate(&["--seeds", "1:2", "--label", "pair", "--manifest", "build/pair.json"]);
    p.ok(&args);
    let first = std::fs::read(p.path("build/pair.json")).unwrap();
    std::fs::write(p.path("build/pair.json"), r#"{"schema": "camdl.output-manifest/v1", "results": []}"#).unwrap();
    let out = p.run(&args);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cached 2 of 2 cells"), "the rerun is a cache hit");
    assert_eq!(std::fs::read(p.path("build/pair.json")).unwrap(), first, "the cache hit rewrote the manifest's bytes");
    let out = p.run_in("other-store", &args);
    assert!(out.status.success());
    assert_eq!(std::fs::read(p.path("build/pair.json")).unwrap(), first, "another store root writes the same manifest");
}

/// A sweep crossed with seeds lists every cell, in the order the invocation
/// asked for them, after the compiled model.
#[test]
fn a_batch_manifest_lists_every_member() {
    let p = Project::new();
    p.ok(&sweep(&["--sweep", "beta=lin(0.2,0.4,3)", "--seeds", "1:2", "--manifest", "build/grid.json"]));
    let m = p.manifest("build/grid.json");
    assert_eq!(kinds(&m), ["compiled_model", "sim", "sim", "sim", "sim", "sim", "sim"]);
    let ids: std::collections::BTreeSet<&str> = results(&m).iter().map(|r| r["run_id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), 7, "every cell is a distinct result");
    let points: Vec<String> = results(&m)[1..].iter().map(|r| r["labels"]["point"].to_string()).collect();
    assert_eq!(points.iter().collect::<std::collections::BTreeSet<_>>().len(), 3, "three sweep points: {points:?}");
}

/// A command that fails removes the manifest an earlier run left at its
/// path and writes none: here a cell the engine cannot run.
#[test]
fn a_failed_command_leaves_no_manifest() {
    let p = Project::new();
    p.ok(&simulate(&["--seed", "1", "--manifest", "build/m.json"]));
    assert!(p.path("build/m.json").exists());
    let out = p.run(&["simulate", "sir.ir.json", "--param", "beta=0.3", "--manifest", "build/m.json"]);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!p.path("build/m.json").exists(), "the failed command left the earlier manifest in place");
    let leftovers: Vec<_> = std::fs::read_dir(p.path("build")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(leftovers.is_empty(), "nothing is left in the manifest's directory: {leftovers:?}");
}

/// A batch in which one member fails writes no manifest, although the
/// members that succeeded are stored.
#[test]
fn a_batch_with_a_failed_member_writes_no_manifest() {
    let p = Project::new();
    p.ok(&simulate(&["--seed", "1", "--manifest", "build/m.json"]));
    // The second design point puts beta outside its declared range, so that
    // cell is refused while the first is simulated and stored.
    let out = p.run(&sweep(&["--sweep", "beta=0.3,50", "--manifest", "build/m.json"]));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("1 of 2 cell(s) did not produce what was asked"), "one member failed: {stderr}");
    let stored: Vec<_> = run_records(&p.path("store")).into_values().filter(|r| r["status"] == "completed").collect();
    assert!(stored.len() >= 2, "the compiled model and the cell that ran are stored");
    assert!(!p.path("build/m.json").exists(), "a batch with a failed member left a manifest");
}

/// A file at the manifest path that is not an output manifest, here the
/// command's own data file, is refused before anything runs and left as it
/// was.
#[test]
fn a_file_that_is_not_a_manifest_is_refused_and_kept() {
    let p = Project::new();
    let before = std::fs::read(p.path("cases.tsv")).unwrap();
    let out = p.run(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--particles", "30", "--manifest", "cases.tsv",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("cases.tsv holds a file that is not an output manifest"), "{stderr}");
    assert_eq!(std::fs::read(p.path("cases.tsv")).unwrap(), before, "the data file is left as it was");
    assert!(!p.path("store").exists(), "nothing is stored");
    // A command refused for another reason leaves the file too.
    let out = p.run(&simulate(&["--param", "nosuch=1", "--manifest", "cases.tsv"]));
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(std::fs::read(p.path("cases.tsv")).unwrap(), before);
}

/// A manifest path that is a directory, or whose directory does not exist,
/// is refused before anything is stored, and `--manifest` with an option
/// that stores nothing kept is refused by argument parsing.
#[test]
fn a_manifest_path_is_refused_before_the_store() {
    let p = Project::new();
    let out = p.run(&simulate(&["--manifest", "build"]));
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("is a directory"));
    let out = p.run(&simulate(&["--manifest", "nowhere/m.json"]));
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("does not exist"));
    assert!(!p.path("store").exists(), "a refused manifest path stores nothing");
    let out = p.run(&simulate(&["--dry-run", "--manifest", "build/m.json"]));
    assert_eq!(out.status.code(), Some(2), "--dry-run stores nothing");
    let out = p.run(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--no-store", "--manifest", "build/m.json",
    ]);
    assert_eq!(out.status.code(), Some(2), "--no-store keeps nothing a manifest could name");
}

/// A filter run lists its compiled model, its data table, the run and each
/// by-product asked for, each by-product naming the run as its upstream.
#[test]
fn manifest_lists_by_products_with_their_upstream() {
    let p = Project::new();
    p.ok(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--particles", "30", "--dt", "1", "--seed", "1", "--save-filtered-streams", "fs.tsv", "--pf-health",
        "health.tsv", "--manifest", "build/pf.json",
    ]);
    let m = p.manifest("build/pf.json");
    let k = kinds(&m);
    assert_eq!(k[..3], ["compiled_model", "data_table", "pfilter"], "{k:?}");
    let run = results(&m).iter().find(|r| r["kind"] == "pfilter").unwrap();
    assert!(run["files"]["loglik"]["file"] == "loglik.toml");
    for kind in ["pf_health", "filtered_streams"] {
        let by = results(&m).iter().find(|r| r["kind"] == kind).unwrap_or_else(|| panic!("no {kind} entry: {k:?}"));
        assert_eq!(by["upstream"], run["run_id"], "{kind} names the filter run");
        assert!(by["files"][kind]["file"].as_str().unwrap().ends_with(".tsv"));
    }
    assert_eq!(results(&m).iter().find(|r| r["kind"] == "data_table").unwrap()["files"]["table"]["file"], "table.tsv");
}

/// A data file named by a relative or by an absolute path gives the same
/// manifest: the manifest holds no path, a data table's included.
#[test]
fn a_data_path_spelled_two_ways_gives_one_manifest() {
    let p = Project::new();
    let absolute = p.path("cases.tsv").to_string_lossy().into_owned();
    let filter = |data: &str, to: &str| {
        p.ok(&[
            "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", data,
            "--obs", "cases", "--particles", "30", "--dt", "1", "--seed", "1", "--manifest", to,
        ]);
        std::fs::read_to_string(p.path(to)).unwrap()
    };
    let relative = filter("cases.tsv", "build/relative.json");
    assert_eq!(filter(&absolute, "build/absolute.json"), relative, "two spellings of one data file");
    let text = relative;
    assert!(!text.contains("cases.tsv\""), "no data path in the manifest: {text}");
    let m = p.manifest("build/relative.json");
    let table = results(&m).iter().find(|r| r["kind"] == "data_table").unwrap();
    assert_eq!(table["labels"], serde_json::json!({ "stream": "cases" }));
}

/// A fit for a profile: beta and gamma estimated by NLopt on the ODE.
const PROFILE_FIT: &str = r#"
[model]
camdl = "sir.ir.json"

[data.observations]
cases = "cases.tsv"

[estimate.beta]
bounds = [0.05, 2.0]
start  = 0.3

[estimate.gamma]
bounds = [0.02, 0.9]
start  = 0.1

[fixed]
N0 = 1000

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1

[config]
dt = 1.0
"#;

/// A profile lists its quantities, compiled model and fit problem, then one
/// profile point per grid value, each with its `mle`.
#[test]
fn a_profile_manifest_lists_every_point() {
    let p = Project::new();
    std::fs::write(p.path("profile.toml"), PROFILE_FIT).unwrap();
    p.ok(&[
        "profile", "sir.ir.json", "--fit", "profile.toml", "--sweep", "beta=lin(0.2,0.4,2)", "--algorithm", "nl-sbplx",
        "--backend", "ode", "--manifest", "build/profile.json",
    ]);
    let m = p.manifest("build/profile.json");
    let k = kinds(&m);
    let stored_points: std::collections::BTreeSet<String> = run_records(&p.path("store"))
        .into_iter()
        .filter(|(_, run)| run["files"].get("mle.toml").is_some())
        .map(|(id, _)| id)
        .collect();
    let listed: std::collections::BTreeSet<String> =
        results(&m).iter().filter(|r| r["kind"] == "profile_point").map(|r| r["run_id"].as_str().unwrap().to_string()).collect();
    assert!(stored_points.len() >= 2, "every grid value has a point: {k:?}");
    assert_eq!(listed, stored_points, "the manifest lists every stored profile point");
    assert_eq!(k[..3], ["quantities", "compiled_model", "fit"], "{k:?}");
    for point in results(&m).iter().filter(|r| r["kind"] == "profile_point") {
        assert_eq!(point["files"]["mle"]["file"], "mle.toml");
    }
}

/// A fit run lists its quantities, compiled model, fit problem and run, the
/// run's files by their logical names; a prediction from it lists the
/// prediction alone, and a score of the prediction the score alone.
#[test]
fn fit_predict_and_score_manifests() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json"]);
    let m = p.manifest("build/fit.json");
    assert_eq!(kinds(&m), ["quantities", "compiled_model", "fit", "fit_run"]);
    let run = &results(&m)[3];
    let names: Vec<&str> = run["files"].as_object().unwrap().keys().map(String::as_str).collect();
    for want in ["draws", "chain_0/trace", "chain_1/trace", "sampler_summary", "summary", "method", "fit_state"] {
        assert!(names.contains(&want), "the fit run lists `{want}`: {names:?}");
    }
    assert_eq!(run["files"]["sampler_summary"]["file"], "pgas_summary.json");
    for provenance in ["fit_config", "run_inputs", "fit_meta", "execution"] {
        assert!(!names.contains(&provenance), "the provenance file `{provenance}` is not listed");
    }
    let problem = &results(&m)[2];
    assert!(problem["files"]["problem"]["file"] == "problem.toml");
    assert!(problem["files"]["inputs/cases"].is_object(), "the training lines are `inputs/<source>`: {}", problem["files"]);

    // The same fit again is a cache hit and writes the same bytes.
    let first = std::fs::read(p.path("build/fit.json")).unwrap();
    std::fs::remove_file(p.path("build/fit.json")).unwrap();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json"]);
    assert_eq!(std::fs::read(p.path("build/fit.json")).unwrap(), first, "a reused fit run writes the computed manifest");

    let fit_run_id = run["run_id"].as_str().unwrap().to_string();
    p.ok(&["predict", &fit_run_id, "--mode", "free_forward", "--label", "pred", "--manifest", "build/pred.json"]);
    let m = p.manifest("build/pred.json");
    assert_eq!(kinds(&m), ["prediction"], "the fit run predicted from is a dependency, not a member");
    let pred = &results(&m)[0];
    assert!(pred["files"]["predictive"]["file"] == "predictive.json");
    assert!(pred["files"]["report"]["file"] == "report.json");

    p.ok(&["score", pred["run_id"].as_str().unwrap(), "--data", "cases=cases.tsv", "--manifest", "build/score.json"]);
    let m = p.manifest("build/score.json");
    assert_eq!(kinds(&m), ["data_table", "score"], "the outcome table the score stored, then the score");
    let names: Vec<&str> = results(&m)[1]["files"].as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(names, ["score", "scored", "summary"]);
}

/// The folder of the one result of `kind` under `store`.
fn folder_of(store: &Path, kind: &str) -> PathBuf {
    let mut found = Vec::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "identity.json") {
                let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                if identity["kind"] == kind {
                    found.push(p.parent().unwrap().to_path_buf());
                }
            }
        }
    }
    assert_eq!(found.len(), 1, "one {kind} under {}: {found:?}", store.display());
    found.remove(0)
}

/// `compare --manifest` lists each prediction and score the comparison read
/// or stored, each prediction before its score, when every one sits in one
/// store; inputs from two stores are refused, naming both.
#[test]
fn a_compare_manifest_lists_its_predictions_and_scores() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "a"]);
    p.ok(&["fit", "run", "fit.toml", "--seed", "2", "--label", "b"]);
    let compare = ["compare", "@a", "@b", "--mode", "free_forward", "--in-sample", "--replicates", "2", "--manifest", "build/cmp.json"];
    p.ok(&compare);
    let m = p.manifest("build/cmp.json");
    assert_eq!(kinds(&m), ["prediction", "score"].repeat(4), "two replicates of each input, each prediction before its score");
    let entries = results(&m);
    assert!(entries[0]["files"]["predictive"].is_object() && entries[1]["files"]["scored"].is_object());
    assert_eq!(entries[1]["upstream"], serde_json::Value::Null, "a score is stored inside its prediction, not a by-product");
    let first = std::fs::read(p.path("build/cmp.json")).unwrap();
    p.ok(&compare);
    assert_eq!(std::fs::read(p.path("build/cmp.json")).unwrap(), first, "a comparison of stored scores writes the same manifest");

    let out = p.run_in("other-store", &["fit", "run", "fit.toml", "--seed", "3"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let other = folder_of(&p.path("other-store"), "fit_run");
    let other = other.to_string_lossy().into_owned();
    let out = p.run(&["compare", "@a", &other, "--mode", "free_forward", "--in-sample", "--manifest", "build/two.json"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("sit in 2 run stores") && stderr.contains("other-store"), "{stderr}");
    assert!(!p.path("build/two.json").exists());
}

/// A fit run swept over a fixed parameter lists every fit run, each after
/// the fit problem it is a run of.
#[test]
fn a_swept_fit_lists_every_run() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--sweep", "gamma=0.1,0.2", "--manifest", "build/sweep.json"]);
    let m = p.manifest("build/sweep.json");
    let k = kinds(&m);
    assert_eq!(k.iter().filter(|k| **k == "fit_run").count(), 2, "{k:?}");
    assert_eq!(k.iter().filter(|k| **k == "fit").count(), 2, "each swept value poses its own problem: {k:?}");
}

/// A filter design scored on several threads lists its points in file
/// order, and gives the same bytes when every point is a cache hit and when
/// every point is computed again in another store.
#[test]
fn a_parallel_design_manifest_is_in_file_order() {
    let p = Project::new();
    let mut design = String::from("beta\n");
    for i in 0..16 {
        design.push_str(&format!("{}\n", 0.2 + 0.01 * i as f64));
    }
    std::fs::write(p.path("design.tsv"), design).unwrap();
    let args = [
        "pfilter", "sir.ir.json", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv", "--obs", "cases",
        "--particles", "40", "--dt", "1", "--seed", "1", "--sweep-file", "design.tsv", "--parallel", "4", "--manifest",
        "build/design.json",
    ];
    p.ok(&args);
    let first = std::fs::read(p.path("build/design.json")).unwrap();
    let m = p.manifest("build/design.json");
    let rows: Vec<u64> = results(&m)
        .iter()
        .filter(|r| r["kind"] == "pfilter")
        .map(|r| r["labels"]["point"]["row"].as_u64().unwrap_or_else(|| panic!("a design label: {}", r["labels"])))
        .collect();
    assert_eq!(rows, (0..16).collect::<Vec<u64>>(), "the points in file order");
    for _ in 0..2 {
        p.ok(&args);
        assert_eq!(std::fs::read(p.path("build/design.json")).unwrap(), first, "a cache hit lists the points in the same order");
    }
    let out = p.run_in("other-store", &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read(p.path("build/design.json")).unwrap(), first, "a recomputation lists them in the same order");
}
