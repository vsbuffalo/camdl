//! `camdl path` and `camdl cat` over handles and output manifests (run spec
//! §4.5): every kind's files through every handle form, the refusals, the
//! `--each` tables and their order, the manifest check, and a workflow that
//! runs a command with `--manifest` and then reads each file it names.
//!
//! Drives the release binary on a three-compartment SIR model; each test has
//! a working directory and run store of its own.

use std::collections::BTreeMap;
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

fn model(compartments: &str, transitions: &str, init: &str) -> String {
    format!(
        r#"
time_unit = 'days
compartments {{ {compartments} }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
{transitions}
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ {init} }}
simulate {{ from = 0 'days  to = 6 'days }}
"#
    )
}

fn sir() -> String {
    model("S, I, R", "  infection : S --> I @ beta * S * I / N0\n  recovery  : I --> R @ gamma * I", "S = 999  I = 1")
}

/// The SIR model with a compartment `D` that nobody enters: the same
/// parameters, and a trajectory with one more column.
fn sird() -> String {
    model("S, I, R, D", "  infection : S --> I @ beta * S * I / N0\n  recovery  : I --> R @ gamma * I", "S = 999  I = 1  D = 0")
}

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

/// A working directory holding the compiled models, their data and the fit
/// configs, with the run store `store/` under it.
struct Project {
    tmp: tempfile::TempDir,
}

impl Project {
    fn new() -> Project {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        for (name, source) in [("sir", sir()), ("sird", sird())] {
            let src = dir.join(format!("{name}.camdl"));
            std::fs::write(&src, source).unwrap();
            let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
            assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(format!("{name}.ir.json")), &out.stdout).unwrap();
        }
        std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
        std::fs::write(dir.join("fit.toml"), FIT).unwrap();
        std::fs::write(dir.join("profile.toml"), PROFILE_FIT).unwrap();
        std::fs::create_dir(dir.join("build")).unwrap();
        Project { tmp }
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir().join(rel)
    }

    fn store(&self) -> PathBuf {
        self.path("store")
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(camdl_bin());
        cmd.env_remove("CLICOLOR_FORCE")
            .env("CAMDL_STORE", self.store())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc_bin())
            .current_dir(self.dir());
        cmd
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command().args(args).output().expect("spawn camdl")
    }

    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(out.status.success(), "camdl {args:?} exited {:?}:\n{}", out.status.code(), String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    }

    /// The stderr of `camdl args`, which must exit 1 and print nothing on
    /// stdout.
    fn refused(&self, args: &[&str]) -> String {
        let out = self.run(args);
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(out.status.code(), Some(1), "camdl {args:?} is refused with exit 1:\n{err}");
        assert!(out.stdout.is_empty(), "camdl {args:?} prints nothing on stdout: {}", String::from_utf8_lossy(&out.stdout));
        err
    }

    /// The one path `camdl path args` prints.
    fn path_of(&self, args: &[&str]) -> PathBuf {
        let out = self.ok(&[&["path"], args].concat());
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1, "camdl path {args:?} prints one line:\n{out}");
        PathBuf::from(lines[0])
    }

    fn manifest(&self, rel: &str) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))).unwrap()
    }

    /// Every stored result's folder, by run id.
    fn folders(&self) -> BTreeMap<String, PathBuf> {
        let mut out = BTreeMap::new();
        let mut stack = vec![self.store()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|n| n == "identity.json") {
                    let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                    out.insert(identity["run_id"].as_str().unwrap().to_string(), p.parent().unwrap().to_path_buf());
                }
            }
        }
        out
    }
}

fn results(m: &serde_json::Value) -> &Vec<serde_json::Value> {
    m["results"].as_array().expect("results is an array")
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

const SIR_PARAMS: [&str; 4] = ["--param", "gamma=0.1", "--param", "N0=1000"];

/// `camdl simulate` of `model` with gamma and N0 fixed, plus `extra`.
fn simulate<'a>(model: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["simulate", model];
    v.extend_from_slice(&SIR_PARAMS);
    v.extend_from_slice(extra);
    v
}

/// Stores one of every kind with a manifest each: a cell with its
/// observations, a filter run with its health table, a fit run, a
/// prediction from it, a score of the prediction and a profile. Returns the
/// manifests' paths.
fn every_kind(p: &Project) -> Vec<&'static str> {
    p.ok(&simulate("sir.ir.json", &["--param", "beta=0.3", "--seed", "1", "--obs-dir", "obs", "--label", "cell", "--manifest", "build/sim.json"]));
    p.ok(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--particles", "20", "--dt", "1", "--seed", "1", "--pf-health", "health.tsv", "--label", "pf",
        "--manifest", "build/pf.json",
    ]);
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json"]);
    let fit = p.manifest("build/fit.json");
    let run = results(&fit).iter().find(|r| r["kind"] == "fit_run").unwrap()["run_id"].as_str().unwrap().to_string();
    p.ok(&["predict", &run, "--mode", "free_forward", "--n-draws", "2", "--label", "pred", "--manifest", "build/pred.json"]);
    let pred = p.manifest("build/pred.json");
    let pred_id = results(&pred)[0]["run_id"].as_str().unwrap().to_string();
    p.ok(&["score", &pred_id, "--data", "cases=cases.tsv", "--label", "sc", "--manifest", "build/score.json"]);
    p.ok(&[
        "profile", "sir.ir.json", "--fit", "profile.toml", "--sweep", "beta=lin(0.2,0.4,2)", "--algorithm", "nl-sbplx",
        "--backend", "ode", "--manifest", "build/profile.json",
    ]);
    vec!["build/sim.json", "build/pf.json", "build/fit.json", "build/pred.json", "build/score.json", "build/profile.json"]
}

/// Every result file a manifest lists is found by `camdl path` through the
/// manifest, the full run id, a 12-digit prefix with the stored name, the
/// kind-qualified logical name and the result's folder, and each path names
/// the file whose digest the manifest records.
#[test]
fn path_names_every_kind_through_every_handle_form() {
    let p = Project::new();
    let manifests = every_kind(&p);
    let folders = p.folders();
    let mut kinds = std::collections::BTreeSet::new();
    for m in manifests {
        let manifest = p.manifest(m);
        for r in results(&manifest) {
            let id = r["run_id"].as_str().unwrap();
            let kind = r["kind"].as_str().unwrap();
            kinds.insert(kind.to_string());
            let folder = folders[id].display().to_string();
            for (logical, f) in r["files"].as_object().unwrap() {
                let stored = f["file"].as_str().unwrap();
                let want = folders[id].join(stored);
                assert!(want.is_absolute());
                let qualified = format!("{kind}:{logical}");
                for args in [
                    vec!["--manifest", m, "--run", &id[..12], logical.as_str()],
                    vec![id, qualified.as_str()],
                    vec![&id[..12], stored],
                    vec![folder.as_str(), logical.as_str()],
                ] {
                    // A folder names the store it lies in, by its canonical path.
                    let got = p.path_of(&args);
                    assert!(got.is_absolute(), "{}", got.display());
                    assert_eq!(got.canonicalize().unwrap(), want.canonicalize().unwrap(), "camdl path {args:?}");
                }
                assert_eq!(sha256_hex(&std::fs::read(&want).unwrap()), f["sha256"].as_str().unwrap(), "{m} {kind} {logical}");
            }
        }
    }
    let want: std::collections::BTreeSet<String> =
        ["compiled_model", "data_table", "quantities", "sim", "obs", "pfilter", "pf_health", "fit", "fit_run", "prediction", "score", "profile_point"]
            .into_iter()
            .map(String::from)
            .collect();
    assert_eq!(kinds, want, "every kind is looked up");

    // A label names the results of its invocation; a bare name picks the one
    // that holds it, and the line naming the result goes to stderr.
    let m = p.manifest("build/fit.json");
    let run = results(&m).iter().find(|r| r["kind"] == "fit_run").unwrap();
    let draws = folders[run["run_id"].as_str().unwrap()].join("draws.tsv");
    assert_eq!(p.path_of(&["@f", "draws"]), draws);
    let out = p.run(&["path", "@f", "draws"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains(&format!("@f → run {}", &run["run_id"].as_str().unwrap()[..6])));
    for (label, name, m) in [("@cell", "traj", "build/sim.json"), ("@pf", "pf_health", "build/pf.json"), ("@pred", "report", "build/pred.json"), ("@sc", "summary", "build/score.json")] {
        let manifest = p.manifest(m);
        let holder = results(&manifest).iter().find(|r| r["files"].get(name).is_some()).unwrap();
        let want = folders[holder["run_id"].as_str().unwrap()].join(holder["files"][name]["file"].as_str().unwrap());
        assert_eq!(p.path_of(&[label, name]), want, "{label} {name}");
    }
    // `@L~1` names the invocation the label held before.
    p.ok(&["fit", "run", "fit.toml", "--seed", "2", "--label", "f"]);
    assert_eq!(p.path_of(&["@f~1", "draws"]), draws, "@f~1 is the first fit");
    assert_ne!(p.path_of(&["@f", "draws"]), draws, "@f is the second");
}

/// Several names print one path per line, in the order named; `--all`
/// prints every result file of one result, by logical name, in name order.
#[test]
fn path_prints_several_names_in_order_and_all_files() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json"]);
    let m = p.manifest("build/fit.json");
    let run = results(&m).iter().find(|r| r["kind"] == "fit_run").unwrap();
    let folder = p.folders()[run["run_id"].as_str().unwrap()].clone();
    let out = p.ok(&["path", "--manifest", "build/fit.json", "chain_1/trace", "draws", "chain_0/trace"]);
    let want: Vec<String> = ["chain_1/trace.tsv", "draws.tsv", "chain_0/trace.tsv"].iter().map(|f| folder.join(f).display().to_string()).collect();
    assert_eq!(out.lines().collect::<Vec<_>>(), want);

    let all = p.ok(&["path", "--manifest", "build/fit.json", "--kind", "fit_run", "--all"]);
    let files = run["files"].as_object().unwrap();
    let want: Vec<String> = files.iter().map(|(l, f)| format!("{l}\t{}", folder.join(f["file"].as_str().unwrap()).display())).collect();
    assert_eq!(all.lines().collect::<Vec<_>>(), want, "every result file, by logical name in name order");
    assert_eq!(p.ok(&["path", "@f", "--all"]), all, "a label names the fit run of its `fit run`");
    let err = p.refused(&["path", "--manifest", "build/fit.json", "--all"]);
    assert!(err.contains("names 4 results, where one is needed") && err.contains("--kind"), "{err}");
}

/// A name is refused, naming what exists: a file the result's method does
/// not write, a bare name several results hold, a kind camdl does not
/// store, `--each` given a handle that names one result, and a handle that
/// names nothing.
#[test]
fn refusals_name_what_exists() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--manifest", "build/fit.json"]);
    let m = p.manifest("build/fit.json");
    let run = results(&m).iter().find(|r| r["kind"] == "fit_run").unwrap()["run_id"].as_str().unwrap().to_string();
    let err = p.refused(&["path", &run, "mle_params"]);
    assert!(err.contains(&format!("the pgas run {} has no `mle_params`", &run[..12])), "{err}");
    assert!(err.contains("draws") && err.contains("chain_0/trace"), "the run's files are listed: {err}");

    p.ok(&simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,3)", "--seed", "1", "--manifest", "build/grid.json"]));
    let err = p.refused(&["path", "--manifest", "build/grid.json", "traj"]);
    assert!(err.contains("3 results of build/grid.json have `traj`") && err.contains("--each for all"), "{err}");
    let err = p.refused(&["cat", "--manifest", "build/grid.json", "traj"]);
    assert!(err.contains("3 results of build/grid.json have `traj`"), "cat refuses as path does: {err}");

    let err = p.refused(&["path", "--manifest", "build/grid.json", "--kind", "fit_runs", "traj"]);
    assert!(err.contains("--kind fit_runs: not a kind"), "{err}");
    let err = p.refused(&["path", "--manifest", "build/grid.json", "--kind", "fit_run", "traj"]);
    assert!(err.contains("--kind fit_run keeps none of the results"), "{err}");
    let err = p.refused(&["path", &run, "draws", "--each"]);
    assert!(err.contains("names one result"), "{err}");
    let out = p.run(&["path", "--manifest", "build/grid.json", "traj", "t", "--each"]);
    assert_eq!(out.status.code(), Some(2), "--each with two names is a usage error");
    let err = p.refused(&["path", "@nothing", "traj"]);
    assert!(err.contains("no run matches '@nothing'"), "{err}");
    let err = p.refused(&["path", "--manifest", "cases.tsv", "traj"]);
    assert!(err.contains("not an output manifest"), "{err}");
    let mut newer = p.manifest("build/grid.json");
    newer["schema"] = "camdl.output-manifest/v2".into();
    std::fs::write(p.path("build/newer.json"), serde_json::to_vec(&newer).unwrap()).unwrap();
    let err = p.refused(&["path", "--manifest", "build/newer.json", "traj"]);
    assert!(err.contains("camdl.output-manifest/v2") && err.contains("reads `camdl.output-manifest/v1`"), "{err}");

    // Read against another store, a manifest fails on its first run id.
    let out = p.command().env("CAMDL_STORE", p.path("other")).args(["path", "--manifest", "build/grid.json", "traj", "--each"]).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no run store at") && err.contains("--store"), "{err}");
}

/// `path --each` over a sweep crossed with seeds: one row per cell, in the
/// manifest's order, with `run_id`, the cell's labels, the swept parameter
/// and `path`; the same bytes on every call and after a cache hit.
#[test]
fn path_each_over_a_sweep_is_in_manifest_order() {
    let p = Project::new();
    let args = simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,3)", "--seeds", "1:2", "--label", "grid", "--manifest", "build/grid.json"]);
    p.ok(&args);
    let m = p.manifest("build/grid.json");
    let table = p.ok(&["path", "--manifest", "build/grid.json", "traj", "--each"]);
    let mut lines = table.lines();
    assert_eq!(
        lines.next().unwrap(),
        "run_id\tmodel\tparams\tscenario\treplicate\tseed\tseed_replicate\tprocess_seed\tpoint_id\tbeta\tpath"
    );
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    let cells: Vec<&serde_json::Value> = results(&m).iter().filter(|r| r["kind"] == "sim").collect();
    assert_eq!(rows.len(), 6);
    let folders = p.folders();
    for (row, cell) in rows.iter().zip(&cells) {
        let id = cell["run_id"].as_str().unwrap();
        assert_eq!(row[0], id, "rows in the manifest's order, by the full run id");
        assert_eq!(row[5], cell["labels"]["seed"].to_string());
        assert_eq!(row[8], cell["labels"]["point"]["point_id"].to_string());
        assert_eq!(row[9].parse::<f64>().unwrap(), cell["labels"]["point"]["values"][0][1].as_f64().unwrap());
        assert_eq!(PathBuf::from(row[10]), folders[id].join("traj.tsv"));
    }
    assert_eq!(p.ok(&["path", "--manifest", "build/grid.json", "traj", "--each"]), table, "the same bytes again");
    assert_eq!(p.ok(&["path", "@grid", "traj", "--each"]), table, "a label lists its invocation's cells in the same order");
    p.ok(&args);
    assert_eq!(p.ok(&["path", "--manifest", "build/grid.json", "traj", "--each"]), table, "the same bytes after a cache hit");
}

/// `path --each` over a swept fit: one row per fit run, in the manifest's
/// order.
#[test]
fn path_each_over_a_swept_fit_is_in_manifest_order() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--sweep", "gamma=0.1,0.2", "--manifest", "build/sweep.json"]);
    let m = p.manifest("build/sweep.json");
    let runs: Vec<&str> = results(&m).iter().filter(|r| r["kind"] == "fit_run").map(|r| r["run_id"].as_str().unwrap()).collect();
    assert_eq!(runs.len(), 2);
    let table = p.ok(&["path", "--manifest", "build/sweep.json", "draws", "--each"]);
    let header: Vec<&str> = table.lines().next().unwrap().split('\t').collect();
    assert_eq!((header[0], *header.last().unwrap()), ("run_id", "path"));
    assert!(header.contains(&"method") && header.contains(&"seed"), "{header:?}");
    let ids: Vec<&str> = table.lines().skip(1).map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(ids, runs, "the fit runs in the manifest's order");
    assert_eq!(p.ok(&["path", "--manifest", "build/sweep.json", "draws", "--each"]), table);
}

/// The label columns are the union of the results' label keys, in the order
/// first seen, with an empty cell where a result lacks a key.
#[test]
fn path_each_takes_the_union_of_label_keys() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--manifest", "build/fit.json"]);
    let table = p.ok(&["path", "--manifest", "build/fit.json", "run.json", "--each"]);
    let mut lines = table.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header, ["run_id", "name", "model", "method", "quantities", "seed", "path"]);
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 4, "quantities, compiled model, fit problem, fit run");
    assert_eq!(rows[0][1], "sir", "the quantities' name");
    assert_eq!(rows[0][3], "", "the quantities have no method");
    assert_eq!(rows[3][3], "pgas", "the run's method");
    assert!(rows.iter().all(|r| r[6].ends_with("/run.json")));
}

/// A manifest that records other bytes than the store holds is refused by
/// `path` and by `cat`, naming the file; a damaged stored file is refused by
/// `cat`.
#[test]
fn a_manifest_for_other_bytes_is_refused() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--param", "beta=0.3", "--seed", "1", "--manifest", "build/sim.json"]));
    let good = p.path_of(&["--manifest", "build/sim.json", "traj"]);
    let mut m = p.manifest("build/sim.json");
    let sim = results(&m).iter().position(|r| r["kind"] == "sim").unwrap();
    m["results"][sim]["files"]["traj"]["sha256"] = sha256_hex(b"other bytes").into();
    std::fs::write(p.path("build/other.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    for cmd in ["path", "cat"] {
        let err = p.refused(&[cmd, "--manifest", "build/other.json", "traj"]);
        assert!(err.contains("build/other.json does not describe the stored file `traj`"), "{cmd}: {err}");
        assert!(err.contains(&sha256_hex(b"other bytes")), "{cmd} names both digests: {err}");
        let err = p.refused(&[cmd, "--manifest", "build/other.json", "traj", "--each"]);
        assert!(err.contains("does not describe"), "{cmd} --each: {err}");
    }
    let mut m = p.manifest("build/sim.json");
    let bytes = m["results"][sim]["files"]["traj"]["bytes"].as_u64().unwrap();
    m["results"][sim]["files"]["traj"]["bytes"] = (bytes + 1).into();
    std::fs::write(p.path("build/longer.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let err = p.refused(&["path", "--manifest", "build/longer.json", "traj"]);
    assert!(err.contains("does not describe") && err.contains(&format!("{bytes} bytes")), "{err}");

    // The file itself changed: cat checks the bytes it prints.
    let mut text = std::fs::read(&good).unwrap();
    text.extend_from_slice(b"7\t0\t7\t0\t0\t0\t0\t0\n");
    std::fs::write(&good, text).unwrap();
    let err = p.refused(&["cat", "--manifest", "build/sim.json", "traj"]);
    assert!(err.contains("damaged"), "{err}");
}

/// `cat` prints the bytes of the file `path` names, for every way of naming
/// it: no name (the kind's summary file), a logical name, a stored name, a
/// qualified name, a by-product's file through its cell, a record, and a
/// prediction's file through the fit run it sits in.
#[test]
fn cat_prints_the_file_path_names() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--param", "beta=0.3", "--seed", "1", "--obs-dir", "obs", "--manifest", "build/sim.json"]));
    let m = p.manifest("build/sim.json");
    let cell = results(&m).iter().find(|r| r["kind"] == "sim").unwrap()["run_id"].as_str().unwrap().to_string();
    let same = |handle: &str, path_args: &[&str], cat_args: &[&str]| {
        let path = p.path_of(&[&[handle], path_args].concat());
        let out = p.run(&[&["cat", handle], cat_args].concat());
        assert!(out.status.success(), "cat {handle} {cat_args:?}: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.stdout, std::fs::read(&path).unwrap(), "cat {handle} {cat_args:?} prints {}", path.display());
    };
    same(&cell, &["traj"], &[]);
    for name in ["traj", "traj.tsv", "sim:traj", "cases", "obs:cases", "cases.tsv", "provenance"] {
        same(&cell, &[name], &[name]);
    }
    same(&cell, &["--kind", "sim", "run.json"], &["--kind", "sim", "run.json"]);

    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json"]);
    let fit = p.manifest("build/fit.json");
    let run = results(&fit).iter().find(|r| r["kind"] == "fit_run").unwrap()["run_id"].as_str().unwrap().to_string();
    same("@f", &["fit_state"], &[]);
    same("@f", &["draws"], &["draws"]);
    p.ok(&["predict", &run, "--mode", "free_forward", "--n-draws", "2", "--manifest", "build/pred.json"]);
    let pred = p.manifest("build/pred.json");
    let folders = p.folders();
    let pred_folder = &folders[results(&pred)[0]["run_id"].as_str().unwrap()];
    let rel = pred_folder.strip_prefix(&folders[&run]).unwrap().display().to_string();
    let nested = format!("{rel}/predictive/cases");
    same(&run, &[&nested], &[&nested]);
    assert_eq!(p.path_of(&[&run, &nested]), pred_folder.join("predictive/cases.tsv"));
}

/// `cat --each` prints every cell's rows under one header, each row after
/// its cell's labels; a JSON file, and files whose headers differ, are
/// refused before a row is printed.
#[test]
fn cat_each_prepends_labels_and_refuses_a_header_mismatch() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,2)", "--seed", "1", "--manifest", "build/sir.json"]));
    let m = p.manifest("build/sir.json");
    let table = p.ok(&["cat", "--manifest", "build/sir.json", "traj", "--each"]);
    let mut lines = table.lines();
    let header = lines.next().unwrap();
    let labels = "model\tparams\tscenario\treplicate\tseed\tseed_replicate\tprocess_seed\tpoint_id\tbeta";
    let folders = p.folders();
    let cells: Vec<&serde_json::Value> = results(&m).iter().filter(|r| r["kind"] == "sim").collect();
    let traj: Vec<String> = cells.iter().map(|c| std::fs::read_to_string(folders[c["run_id"].as_str().unwrap()].join("traj.tsv")).unwrap()).collect();
    let traj_header = traj[0].lines().find(|l| !l.starts_with('#')).unwrap();
    assert_eq!(header, format!("{labels}\t{traj_header}"));
    let rows: Vec<&str> = lines.collect();
    let mut want = Vec::new();
    for (cell, text) in cells.iter().zip(&traj) {
        let beta = cell["labels"]["point"]["values"][0][1].as_f64().unwrap();
        for row in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
            want.push((beta, row.to_string()));
        }
    }
    assert_eq!(rows.len(), want.len(), "every cell's rows");
    for (row, (beta, file_row)) in rows.iter().zip(&want) {
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols[8].parse::<f64>().unwrap(), *beta, "the row keeps its cell's coordinates");
        assert_eq!(cols[9..].join("\t"), *file_row, "then the file's own row");
    }

    let err = p.refused(&["cat", "--manifest", "build/sir.json", "--kind", "compiled_model", "model", "--each"]);
    assert!(err.contains("is JSON; cat --each reads TSV files only"), "{err}");

    // One manifest whose cells come from two models with different
    // compartments.
    p.ok(&simulate("sird.ir.json", &["--param", "beta=0.3", "--seed", "1", "--manifest", "build/sird.json"]));
    let mut both = m.clone();
    let other = p.manifest("build/sird.json");
    both["results"].as_array_mut().unwrap().extend(results(&other).iter().cloned());
    std::fs::write(p.path("build/both.json"), serde_json::to_vec_pretty(&both).unwrap()).unwrap();
    let odd = results(&other).iter().find(|r| r["kind"] == "sim").unwrap()["run_id"].as_str().unwrap().to_string();
    let err = p.refused(&["cat", "--manifest", "build/both.json", "traj", "--each"]);
    assert!(err.contains(&format!("the sim {} has a `traj` whose header differs", &odd[..12])), "{err}");
}

/// A workflow: a shell script runs `simulate --manifest`, then
/// `camdl path --each`, and opens every file the table names; each file is
/// the one whose digest the manifest records.
#[test]
fn a_workflow_reads_each_file_a_manifest_names() {
    let p = Project::new();
    let bin = p.path("bin");
    std::fs::create_dir(&bin).unwrap();
    std::os::unix::fs::symlink(camdl_bin(), bin.join("camdl")).unwrap();
    let script = r#"
set -eu
camdl simulate sir.ir.json --param gamma=0.1 --param N0=1000 --sweep 'beta=lin(0.2,0.4,3)' --seed 1 --manifest build/grid.json 2>/dev/null
camdl path --manifest build/grid.json traj --each > build/cells.tsv
tail -n +2 build/cells.tsv | awk -F '\t' '{ print $1 "\t" $NF }' | while IFS="$(printf '\t')" read -r id path; do
    printf '%s\t%s\n' "$id" "$(wc -l < "$path" | tr -d ' ')"
done > build/lines.tsv
"#;
    let mut sh = Command::new("sh");
    sh.arg("-c").arg(script).current_dir(p.dir()).env("PATH", format!("{}:{}", bin.display(), std::env::var("PATH").unwrap_or_default()));
    sh.env("CAMDL_STORE", p.store()).env("CAMDL_SKIP_VERSION_CHECK", "1").env("CAMDLC", camdlc_bin()).env_remove("CLICOLOR_FORCE");
    let out = sh.output().unwrap();
    assert!(out.status.success(), "the workflow script: {}", String::from_utf8_lossy(&out.stderr));
    let m = p.manifest("build/grid.json");
    let lines = std::fs::read_to_string(p.path("build/lines.tsv")).unwrap();
    let got: Vec<(String, usize)> = lines
        .lines()
        .map(|l| {
            let (id, n) = l.split_once('\t').unwrap();
            (id.to_string(), n.parse().unwrap())
        })
        .collect();
    let folders = p.folders();
    let want: Vec<(String, usize)> = results(&m)
        .iter()
        .filter(|r| r["kind"] == "sim")
        .map(|r| {
            let id = r["run_id"].as_str().unwrap();
            let bytes = std::fs::read(folders[id].join("traj.tsv")).unwrap();
            assert_eq!(sha256_hex(&bytes), r["files"]["traj"]["sha256"].as_str().unwrap());
            (id.to_string(), bytes.iter().filter(|b| **b == b'\n').count())
        })
        .collect();
    assert_eq!(got, want, "the script read each cell's file, in the manifest's order");
}

/// `folder` relative to `from`, through their common ancestor, as `../..`
/// then the rest.
fn relative(from: &Path, folder: &Path) -> String {
    let common = from.ancestors().find(|a| folder.starts_with(a)).unwrap();
    let up = from.strip_prefix(common).unwrap().components().count();
    let mut parts: Vec<String> = vec!["..".into(); up];
    parts.push(folder.strip_prefix(common).unwrap().display().to_string());
    parts.join("/")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let target = to.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}

/// A path through a nested result reaches only a descendant of the
/// candidate: a path that climbs out with `..` is refused, a result whose
/// folder sits inside the candidate but whose identity names another parent
/// is refused, and a manifest names only its own entries, so a prediction
/// made after the manifest is not reached through it.
#[test]
fn a_nested_path_reaches_only_a_descendant() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,2)", "--seed", "1", "--manifest", "build/grid.json"]));
    let m = p.manifest("build/grid.json");
    let cells: Vec<String> = results(&m).iter().filter(|r| r["kind"] == "sim").map(|r| r["run_id"].as_str().unwrap().to_string()).collect();
    let folders = p.folders();
    let (a, b) = (&folders[&cells[0]], &folders[&cells[1]]);
    let climb = format!("{}/traj", relative(a, b));
    assert!(climb.starts_with("../"), "{climb}");
    let err = p.refused(&["path", &cells[0], &climb]);
    assert!(err.contains(&format!("has no `{climb}`")), "a `..` path is not followed: {err}");
    let err = p.refused(&["path", "--manifest", "build/grid.json", "--run", &cells[0][..12], &climb]);
    assert!(err.contains("A manifest names only its own entries"), "{err}");


    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--manifest", "build/fit.json"]);
    let fit = p.manifest("build/fit.json");
    let run = results(&fit).iter().find(|r| r["kind"] == "fit_run").unwrap()["run_id"].as_str().unwrap().to_string();
    p.ok(&["predict", &run, "--mode", "free_forward", "--n-draws", "2", "--manifest", "build/pred.json"]);
    let pred = p.manifest("build/pred.json");
    let folders = p.folders();
    let pred_folder = &folders[results(&pred)[0]["run_id"].as_str().unwrap()];
    let nested = format!("{}/report", pred_folder.strip_prefix(&folders[&run]).unwrap().display());
    assert_eq!(p.path_of(&[&run, &nested]), pred_folder.join("report.json"), "a handle reaches the run's prediction");
    let err = p.refused(&["path", "--manifest", "build/fit.json", "--kind", "fit_run", &nested]);
    assert!(err.contains("A manifest names only its own entries"), "{err}");
    let err = p.refused(&["cat", "--manifest", "build/fit.json", "--kind", "fit_run", &nested]);
    assert!(err.contains("A manifest names only its own entries"), "cat refuses as path does: {err}");

    // A path that climbs with `..` is refused even where it would land on a
    // descendant.
    let dotted = format!("predict/../{}", nested);
    let err = p.refused(&["path", &run, &dotted]);
    assert!(err.contains(&format!("has no `{dotted}`")), "{err}");

    // Cell b's folder copied inside the run's prediction: a folder holding
    // an identity record, below the run, whose identity names no parent.
    // The store checks a result's folder down to the first nested result, so
    // the run itself still reads; the path through the copy must not reach
    // cell b's stored file.
    let inside = pred_folder.strip_prefix(&folders[&run]).unwrap().join("inside");
    copy_dir(b, &folders[&run].join(&inside));
    assert!(p.path_of(&[&run, "draws"]).ends_with("draws.tsv"), "the run still reads");
    let through = format!("{}/traj", inside.display());
    let err = p.refused(&["path", &run, &through]);
    assert!(err.contains(&format!("has no `{through}`")), "a result that is not a descendant is not reached: {err}");
}

/// A listed result the store cannot read blocks only the lookups it could
/// answer: a filter run's own file and `show` read past its unreadable
/// by-product, which is refused by name; a sweep's other cells are read past
/// an unreadable cell, and `--each` over all of them is refused.
#[test]
fn an_unreadable_member_blocks_only_the_lookups_it_could_answer() {
    let p = Project::new();
    p.ok(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--particles", "20", "--dt", "1", "--seed", "1", "--pf-health", "health.tsv", "--label", "pf",
        "--manifest", "build/pf.json",
    ]);
    let m = p.manifest("build/pf.json");
    let id_of = |kind: &str| results(&m).iter().find(|r| r["kind"] == kind).unwrap()["run_id"].as_str().unwrap().to_string();
    let folders = p.folders();
    std::fs::remove_file(folders[&id_of("pf_health")].join("run.json")).unwrap();
    let run = folders[&id_of("pfilter")].clone();
    assert!(p.ok(&["cat", "@pf"]).starts_with("loglik"), "the run's default file");
    assert_eq!(p.path_of(&["@pf", "loglik"]), run.join("loglik.toml"));
    assert_eq!(p.path_of(&[&id_of("pfilter"), "loglik"]), run.join("loglik.toml"));
    assert!(p.ok(&["show", "@pf"]).contains("pfilter"));
    let err = p.refused(&["path", "@pf", "pf_health"]);
    assert!(err.contains(&format!("The pf_health {} is not read", &id_of("pf_health")[..12])) && err.contains("--pf-health and --force"), "{err}");
    let err = p.refused(&["path", &id_of("pfilter"), "pf_health"]);
    assert!(err.contains("is not read") && err.contains("--pf-health"), "the run's handle names its unreadable by-product: {err}");

    p.ok(&simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,3)", "--seed", "1", "--label", "grid", "--manifest", "build/grid.json"]));
    let g = p.manifest("build/grid.json");
    let cells: Vec<String> = results(&g).iter().filter(|r| r["kind"] == "sim").map(|r| r["run_id"].as_str().unwrap().to_string()).collect();
    let folders = p.folders();
    std::fs::remove_file(folders[&cells[1]].join("run.json")).unwrap();
    assert_eq!(p.path_of(&["@grid", "--run", &cells[0][..12], "traj"]), folders[&cells[0]].join("traj.tsv"));
    assert_eq!(p.path_of(&["@grid", "--run", &cells[2][..12], "traj"]), folders[&cells[2]].join("traj.tsv"));
    assert!(p.path_of(&["@grid", "model"]).ends_with("model.ir.json"), "the compiled model is read past the cell");
    let err = p.refused(&["path", "@grid", "traj", "--each"]);
    assert!(err.contains(&format!("The sim {} is not read", &cells[1][..12])), "--each never leaves a holder out: {err}");
    let err = p.refused(&["path", "@grid", "--run", &cells[1][..12], "traj"]);
    assert!(err.contains("is not read"), "the unreadable cell named alone: {err}");
}

/// A run id's own result answers before its by-products: the cell's
/// `run.json`, not its observations'; a by-product's file stays reachable
/// through the cell where the cell has no such name.
#[test]
fn a_run_ids_own_result_answers_first() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--param", "beta=0.3", "--seed", "1", "--obs-dir", "obs", "--manifest", "build/sim.json"]));
    let m = p.manifest("build/sim.json");
    let id_of = |kind: &str| results(&m).iter().find(|r| r["kind"] == kind).unwrap()["run_id"].as_str().unwrap().to_string();
    let folders = p.folders();
    let (cell, obs) = (id_of("sim"), id_of("obs"));
    assert_eq!(p.path_of(&[&cell, "run.json"]), folders[&cell].join("run.json"));
    assert_eq!(p.path_of(&[&cell, "cases"]), folders[&obs].join("cases.tsv"));
    assert_eq!(p.path_of(&[&cell, "obs:run.json"]), folders[&obs].join("run.json"));
    assert_eq!(p.path_of(&[&obs, "run.json"]), folders[&obs].join("run.json"), "a by-product's own id names it first");
}

/// The other refusals say what to do: the removed `cat --stream`, a kind no
/// result is, a manifest read where no store is, and a multi-cell label
/// given to `show`; a label names the kind of the result it resolved to.
#[test]
fn refusals_say_what_to_write() {
    let p = Project::new();
    p.ok(&simulate("sir.ir.json", &["--sweep", "beta=lin(0.2,0.4,2)", "--seed", "1", "--label", "grid", "--manifest", "build/grid.json"]));
    let m = p.manifest("build/grid.json");
    let cell = results(&m).iter().find(|r| r["kind"] == "sim").unwrap()["run_id"].as_str().unwrap().to_string();
    let err = p.refused(&["cat", &cell, "--stream", "traj"]);
    assert!(err.contains("write `camdl cat HANDLE traj`"), "{err}");
    let err = p.refused(&["path", "--manifest", "build/grid.json", "fit_run:draws"]);
    assert!(err.contains("is a `fit_run`") && err.contains("compiled_model, sim"), "{err}");
    let out = p.command().env("CAMDL_STORE", p.path("nowhere")).args(["path", "--manifest", "build/grid.json", "traj", "--each"]).output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1));
    assert!(err.contains("no run store at") && err.contains("--store"), "{err}");
    let err = p.refused(&["show", "@grid"]);
    assert!(err.contains("`camdl cat @grid` renders their table"), "{err}");

    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--manifest", "build/fit.json"]);
    let fit = p.manifest("build/fit.json");
    let run = results(&fit).iter().find(|r| r["kind"] == "fit_run").unwrap()["run_id"].as_str().unwrap().to_string();
    p.ok(&["predict", &run, "--mode", "free_forward", "--n-draws", "2", "--label", "pred"]);
    let out = p.run(&["path", "@pred", "report"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("@pred → prediction "), "{}", String::from_utf8_lossy(&out.stderr));
}
