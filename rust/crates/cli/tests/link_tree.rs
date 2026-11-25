//! `--link-root DIR` and `CAMDL_LINK_ROOT`: the link tree a command that
//! stores results writes beside its output manifest (run spec §2.11), read as
//! a person browsing results reads it.
//!
//! Drives the release binary on a three-compartment SIR model; each test has
//! a working directory, run store and link root of its own.

use std::collections::{BTreeMap, BTreeSet};
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
simulate { from = 0 'days  to = 6 'days }
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

/// A working directory holding the compiled model, its data and the fit
/// configs, with the run store `store/` under it.
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

    /// `camdl args` in `cwd` (relative to the project), with `env` set and
    /// `CAMDL_LINK_ROOT` unset unless `env` sets it.
    fn run_with(&self, cwd: &str, env: &[(&str, &str)], args: &[&str]) -> Output {
        let mut cmd = Command::new(camdl_bin());
        cmd.env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_LINK_ROOT")
            .env("CAMDL_STORE", self.path("store"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc_bin())
            .current_dir(self.path(cwd));
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.args(args).output().expect("spawn camdl")
    }

    fn run(&self, args: &[&str]) -> Output {
        self.run_with(".", &[], args)
    }

    fn ok_with(&self, cwd: &str, env: &[(&str, &str)], args: &[&str]) -> String {
        let out = self.run_with(cwd, env, args);
        assert!(out.status.success(), "camdl {args:?} exited {:?}:\n{}", out.status.code(), String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    }

    fn ok(&self, args: &[&str]) -> String {
        self.ok_with(".", &[], args)
    }

    /// The stderr of `camdl args`, which must exit 1.
    fn refused(&self, args: &[&str]) -> String {
        let out = self.run(args);
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(out.status.code(), Some(1), "camdl {args:?} is refused with exit 1:\n{err}");
        err
    }

    fn manifest(&self, rel: &str) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))).unwrap()
    }

    /// The one path `camdl path args` prints.
    fn path_of(&self, args: &[&str]) -> PathBuf {
        let out = self.ok(&[&["path"], args].concat());
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1, "camdl path {args:?} prints one line:\n{out}");
        PathBuf::from(lines[0])
    }

    /// The number of invocation records in the store.
    fn invocations(&self) -> usize {
        std::fs::read_dir(self.path("store/.invocations")).map(|d| d.count()).unwrap_or(0)
    }
}

fn results(m: &serde_json::Value) -> &Vec<serde_json::Value> {
    m["results"].as_array().expect("results is an array")
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Every symbolic link under `dir`, by its path relative to `dir`. Links
/// are not followed.
fn links_under(dir: &Path) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            let meta = std::fs::symlink_metadata(&p).unwrap();
            if meta.file_type().is_symlink() {
                out.insert(p.strip_prefix(dir).unwrap().to_path_buf());
            } else if meta.is_dir() {
                stack.push(p);
            }
        }
    }
    out
}

/// The group a primary kind's links sit in; `None` for a kind that gets
/// none of its own (a content artifact, a fit problem or a by-product).
fn group(kind: &str) -> Option<&'static str> {
    match kind {
        "fit_run" => Some("fits"),
        "sim" => Some("sims"),
        "pfilter" => Some("filters"),
        "prediction" => Some("predictions"),
        "profile_point" => Some("profiles"),
        "score" => Some("scores"),
        _ => None,
    }
}

/// Where the layout of run spec §2.11 puts each result file of `manifest`,
/// relative to the link root: (link path, run id, logical name).
fn expected_links(manifest: &serde_json::Value) -> Vec<(PathBuf, String, String)> {
    let label = manifest["label"].as_str();
    let mut per_group: BTreeMap<&str, usize> = BTreeMap::new();
    for r in results(manifest) {
        if let Some(g) = group(r["kind"].as_str().unwrap()) {
            *per_group.entry(g).or_default() += 1;
        }
    }
    let mut dir_of: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut out = Vec::new();
    for r in results(manifest) {
        let id = r["run_id"].as_str().unwrap().to_string();
        let dir = match (group(r["kind"].as_str().unwrap()), r["upstream"].as_str()) {
            (Some(g), None) => {
                let d = match label {
                    Some(l) if per_group[g] == 1 => Path::new(g).join(l),
                    Some(l) => Path::new(g).join(l).join(&id[..12]),
                    None => Path::new(g).join(&id[..12]),
                };
                dir_of.insert(id.clone(), d.clone());
                d
            }
            (None, Some(up)) => dir_of[up].clone(),
            (None, None) => continue,
            (Some(_), Some(_)) => panic!("a primary kind with an upstream: {r}"),
        };
        for (logical, f) in r["files"].as_object().unwrap() {
            let stored = f["file"].as_str().unwrap();
            let name = match Path::new(stored).extension() {
                Some(ext) => format!("{logical}.{}", ext.to_str().unwrap()),
                None => logical.clone(),
            };
            out.push((dir.join(name), id.clone(), logical.clone()));
        }
    }
    out
}

/// Asserts that the links under `root` are exactly those `manifests` give,
/// and that each resolves to the file `camdl path` names for it, with the
/// bytes the manifest records.
fn assert_links_match(p: &Project, root: &Path, manifests: &[&str]) {
    let mut want = BTreeSet::new();
    for m in manifests {
        let manifest = p.manifest(m);
        for (link, id, logical) in expected_links(&manifest) {
            let at = root.join(&link);
            let meta = std::fs::symlink_metadata(&at).unwrap_or_else(|e| panic!("{m}: no link at {}: {e}", at.display()));
            assert!(meta.file_type().is_symlink(), "{} is a symbolic link", at.display());
            let stored = p.path_of(&["--manifest", m, "--run", &id[..12], &logical]);
            assert_eq!(at.canonicalize().unwrap(), stored.canonicalize().unwrap(), "{m}: {} resolves to `camdl path`'s file", link.display());
            let bytes = std::fs::read(&at).unwrap();
            assert_eq!(bytes, std::fs::read(&stored).unwrap());
            let entry = results(&manifest).iter().find(|r| r["run_id"] == id.as_str()).unwrap();
            assert_eq!(sha256_hex(&bytes), entry["files"][&logical]["sha256"].as_str().unwrap(), "{m}: {}", link.display());
            want.insert(link);
        }
    }
    assert_eq!(links_under(root), want, "the link root holds exactly the links of the manifests' results");
}

const SIM: [&str; 8] = ["simulate", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000"];

fn sim<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = SIM.to_vec();
    v.extend_from_slice(extra);
    v
}

/// Each storing command that takes `--label` writes one link per result
/// file of each result it lists, at `<root>/<group>/<label>/<logical>.<ext>`,
/// a by-product's files beside its upstream's, several results of a group
/// one level deeper under their run ids and, without a label, each result
/// under its run id; content artifacts and fit problems get none. Each link
/// resolves to the file `camdl path` names, with the bytes the manifest
/// records.
#[test]
fn links_are_written_for_every_kind_and_resolve_to_the_stored_files() {
    let p = Project::new();
    p.ok(&sim(&["--seed", "1", "--obs-dir", "obs", "--label", "cell", "--manifest", "build/sim.json", "--link-root", "results"]));
    p.ok(&sim(&["--seeds", "1:2", "--label", "pair", "--manifest", "build/pair.json", "--link-root", "results"]));
    p.ok(&[
        "pfilter", "sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--data", "cases.tsv",
        "--obs", "cases", "--particles", "20", "--dt", "1", "--seed", "1", "--pf-health", "health.tsv", "--label", "pf",
        "--manifest", "build/pf.json", "--link-root", "results",
    ]);
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/fit.json", "--link-root", "results"]);
    p.ok(&["predict", "@f", "--mode", "free_forward", "--n-draws", "2", "--label", "pred", "--manifest", "build/pred.json", "--link-root", "results"]);
    let pred = p.manifest("build/pred.json");
    let pred_id = results(&pred)[0]["run_id"].as_str().unwrap().to_string();
    p.ok(&["score", &pred_id, "--data", "cases=cases.tsv", "--label", "sc", "--manifest", "build/score.json", "--link-root", "results"]);
    p.ok(&[
        "profile", "sir.ir.json", "--fit", "profile.toml", "--sweep", "beta=lin(0.2,0.4,2)", "--algorithm", "nl-sbplx",
        "--backend", "ode", "--manifest", "build/profile.json", "--link-root", "results",
    ]);
    let manifests = ["build/sim.json", "build/pair.json", "build/pf.json", "build/fit.json", "build/pred.json", "build/score.json", "build/profile.json"];
    assert_links_match(&p, &p.path("results"), &manifests);

    // The layout, spelled out for the cases the rule above derives.
    let r = p.path("results");
    for link in ["sims/cell/traj.tsv", "sims/cell/cases.tsv", "filters/pf/loglik.toml", "filters/pf/pf_health.tsv", "fits/f/draws.tsv",
        "fits/f/chain_0/trace.tsv", "predictions/pred/report.json", "scores/sc/summary.tsv"]
    {
        assert!(r.join(link).exists(), "{link}");
    }
    let pair: Vec<String> = std::fs::read_dir(r.join("sims/pair")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.')).collect();
    assert_eq!(pair.len(), 2, "two cells of one invocation sit one level deeper, by run id: {pair:?}");
    let profiles: Vec<String> = std::fs::read_dir(r.join("profiles")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.')).collect();
    let points: BTreeSet<String> = results(&p.manifest("build/profile.json")).iter().filter(|r| r["kind"] == "profile_point")
        .map(|r| r["run_id"].as_str().unwrap()[..12].to_string()).collect();
    assert!(points.len() > 1);
    assert_eq!(profiles.into_iter().collect::<BTreeSet<_>>(), points, "without a label each point sits under its run id");
    assert!(!r.join("fits/f/problem.toml").exists() && !r.join("fits/f/model.toml").exists(), "a fit problem gets no links");
}

/// A link's target is relative when the store and the link root share a
/// directory below `/`, so moving the project keeps the links working.
#[test]
fn links_are_relative_under_a_shared_parent() {
    let p = Project::new();
    p.ok(&sim(&["--seed", "1", "--label", "cell", "--link-root", "results"]));
    let link = p.path("results/sims/cell/traj.tsv");
    let target = std::fs::read_link(&link).unwrap();
    assert!(target.is_relative(), "relative target: {}", target.display());
    let before = std::fs::read(&link).unwrap();
    let moved = tempfile::tempdir().unwrap();
    let to = moved.path().join("project");
    std::fs::rename(p.dir(), &to).unwrap();
    assert_eq!(std::fs::read(to.join("results/sims/cell/traj.tsv")).unwrap(), before, "the link resolves after the project moved");
    std::fs::rename(&to, p.dir()).unwrap();
}

/// `--link-root` wins over `CAMDL_LINK_ROOT`, which is the fallback; with
/// neither, no links are written.
#[test]
fn the_flag_wins_over_the_environment_variable() {
    let p = Project::new();
    p.ok_with(".", &[("CAMDL_LINK_ROOT", "from_env")], &sim(&["--seed", "1", "--label", "a"]));
    assert!(p.path("from_env/sims/a/traj.tsv").exists(), "CAMDL_LINK_ROOT is the fallback");
    p.ok_with(".", &[("CAMDL_LINK_ROOT", "from_env")], &sim(&["--seed", "1", "--label", "b", "--link-root", "from_flag"]));
    assert!(p.path("from_flag/sims/b/traj.tsv").exists(), "--link-root is used");
    assert!(!p.path("from_env/sims/b").exists(), "--link-root wins over CAMDL_LINK_ROOT");
    p.ok(&sim(&["--seed", "1", "--label", "c"]));
    assert!(!p.path("from_env/sims/c").exists() && !p.path("from_flag/sims/c").exists(), "neither: no links");
    // An empty variable names no root.
    p.ok_with(".", &[("CAMDL_LINK_ROOT", "")], &sim(&["--seed", "1", "--label", "d"]));
    assert!(!p.path("sims").exists(), "an empty CAMDL_LINK_ROOT writes no links");
}

/// A relative root, given by the flag or the variable, is relative to the
/// directory the command runs in.
#[test]
fn a_relative_root_is_relative_to_the_working_directory() {
    let p = Project::new();
    std::fs::create_dir(p.path("sub")).unwrap();
    let args = ["simulate", "../sir.ir.json", "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000", "--seed", "1"];
    p.ok_with("sub", &[], &[&args[..], &["--label", "a", "--link-root", "results"]].concat());
    assert!(p.path("sub/results/sims/a/traj.tsv").exists());
    p.ok_with("sub", &[("CAMDL_LINK_ROOT", "env_results")], &[&args[..], &["--label", "b"]].concat());
    assert!(p.path("sub/env_results/sims/b/traj.tsv").exists());
    assert!(!p.path("results").exists() && !p.path("env_results").exists());
}

/// A command run again with the label replaces the label's links: they
/// follow the latest successful invocation with it, as `@label` does, and no
/// link of the earlier run survives, in its group or another.
#[test]
fn links_replace_earlier_links() {
    let p = Project::new();
    p.ok(&sim(&["--seeds", "1:2", "--label", "x", "--link-root", "results"]));
    assert_eq!(links_under(&p.path("results")).len(), 2);
    p.ok(&sim(&["--seed", "3", "--label", "x", "--manifest", "build/x.json", "--link-root", "results"]));
    assert_links_match(&p, &p.path("results"), &["build/x.json"]);
    assert_eq!(p.path_of(&["@x", "traj"]).canonicalize().unwrap(), p.path("results/sims/x/traj.tsv").canonicalize().unwrap());

    // The label moved to a fit: the cell's links go with it.
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "x", "--manifest", "build/fx.json", "--link-root", "results"]);
    assert!(!p.path("results/sims/x").exists(), "the label's links in another group are removed");
    assert_links_match(&p, &p.path("results"), &["build/fx.json"]);
}

/// A command that fails leaves no links for its label, though an earlier
/// run with the label had written some.
#[test]
fn a_failing_command_leaves_no_links_for_its_label() {
    let p = Project::new();
    p.ok(&sim(&["--seed", "1", "--label", "x", "--link-root", "results"]));
    assert!(p.path("results/sims/x/traj.tsv").exists());
    p.refused(&sim(&["--seed", "1", "--param", "no_such_parameter=1", "--label", "x", "--link-root", "results"]));
    assert!(!p.path("results/sims/x").exists(), "the failed command removed the label's links");
    assert!(links_under(&p.path("results")).is_empty());
}

/// A file, a foreign symbolic link, or a directory holding anything camdl
/// did not write, at a link directory's path, is refused before anything is
/// stored, and left as it is.
#[test]
fn a_file_or_foreign_link_in_the_way_is_refused_and_kept() {
    let p = Project::new();
    std::fs::create_dir_all(p.path("results/sims")).unwrap();

    // A regular file at the label's directory.
    std::fs::write(p.path("results/sims/x"), "notes\n").unwrap();
    let before = p.invocations();
    let err = p.refused(&sim(&["--seed", "1", "--label", "x", "--link-root", "results"]));
    assert!(err.contains("results/sims/x") && err.contains("not write"), "{err}");
    assert_eq!(std::fs::read(p.path("results/sims/x")).unwrap(), b"notes\n");
    assert_eq!(p.invocations(), before, "nothing was stored");
    std::fs::remove_file(p.path("results/sims/x")).unwrap();

    // A symbolic link at the label's directory.
    std::fs::create_dir(p.path("elsewhere")).unwrap();
    std::os::unix::fs::symlink(p.path("elsewhere"), p.path("results/sims/x")).unwrap();
    let err = p.refused(&sim(&["--seed", "1", "--label", "x", "--link-root", "results"]));
    assert!(err.contains("results/sims/x"), "{err}");
    assert!(std::fs::symlink_metadata(p.path("results/sims/x")).unwrap().file_type().is_symlink());
    std::fs::remove_file(p.path("results/sims/x")).unwrap();

    // A link directory camdl wrote, with a person's file and a foreign link
    // put into it.
    p.ok(&sim(&["--seed", "1", "--label", "x", "--link-root", "results"]));
    std::fs::write(p.path("results/sims/x/my_notes.md"), "mine\n").unwrap();
    let err = p.refused(&sim(&["--seed", "2", "--label", "x", "--link-root", "results"]));
    assert!(err.contains("my_notes.md"), "{err}");
    assert_eq!(std::fs::read(p.path("results/sims/x/my_notes.md")).unwrap(), b"mine\n");
    assert!(p.path("results/sims/x/traj.tsv").exists(), "the refused command removed nothing");
    std::fs::remove_file(p.path("results/sims/x/my_notes.md")).unwrap();
    std::fs::remove_file(p.path("results/sims/x/traj.tsv")).unwrap();
    std::os::unix::fs::symlink(p.path("cases.tsv"), p.path("results/sims/x/traj.tsv")).unwrap();
    let err = p.refused(&sim(&["--seed", "2", "--label", "x", "--link-root", "results"]));
    assert!(err.contains("traj.tsv"), "{err}");
    assert_eq!(std::fs::read_link(p.path("results/sims/x/traj.tsv")).unwrap(), p.path("cases.tsv"));

    // A label the store accepts that is not a plain directory name is
    // refused with a root; `..` is refused by the store's label rule.
    let err = p.refused(&sim(&["--seed", "1", "--label", ".x", "--link-root", "results"]));
    assert!(err.contains("--link-root"), "{err}");
    let err = p.refused(&sim(&["--seed", "1", "--label", "..", "--link-root", "results"]));
    assert!(err.contains("the label `..` is refused"), "{err}");
}

/// Links are an export: a command with them stores and reuses what the
/// command without them does, under the same run ids and digests.
#[test]
fn links_move_no_run_id() {
    let p = Project::new();
    p.ok(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/plain.json"]);
    let out = p.run(&["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--manifest", "build/linked.json", "--link-root", "results"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cache hit"), "the linked run reuses the stored fit: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read(p.path("build/plain.json")).unwrap(), std::fs::read(p.path("build/linked.json")).unwrap(), "one manifest");
    assert!(p.path("results/fits/f/draws.tsv").exists());
}

/// `--link-root` is refused where nothing is kept in the store.
#[test]
fn link_root_is_refused_where_nothing_is_stored() {
    let p = Project::new();
    let out = p.run(&sim(&["--seed", "1", "--dry-run", "--link-root", "results"]));
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    // The variable is ignored there: a dry run touches no link.
    p.ok(&sim(&["--seed", "1", "--label", "x", "--link-root", "results"]));
    p.ok_with(".", &[("CAMDL_LINK_ROOT", "results")], &sim(&["--seed", "1", "--label", "x", "--dry-run"]));
    assert!(p.path("results/sims/x/traj.tsv").exists(), "a dry run leaves the label's links");
}

/// Two labels that differ only in case name one directory on a
/// case-insensitive volume, the default on macOS. The second is refused,
/// naming both, and the first label's links are left as they were. On a
/// case-sensitive filesystem the two directories are distinct and the test
/// checks that; the unit test
/// `a_directory_recorded_for_another_label_is_refused_naming_both` covers
/// the refusal there.
#[test]
fn labels_one_directory_holds_are_refused() {
    let p = Project::new();
    p.ok(&sim(&["--seed", "4", "--label", "Up", "--link-root", "results"]));
    let before = std::fs::read_link(p.path("results/sims/Up/traj.tsv")).unwrap();
    std::fs::write(p.path("Probe"), "").unwrap();
    let case_insensitive = p.path("probe").exists();
    if case_insensitive {
        let err = p.refused(&sim(&["--seed", "5", "--label", "up", "--link-root", "results"]));
        assert!(err.contains("`Up`") && err.contains("`up`"), "{err}");
        assert_eq!(std::fs::read_link(p.path("results/sims/Up/traj.tsv")).unwrap(), before, "Up's links are untouched");
    } else {
        p.ok(&sim(&["--seed", "5", "--label", "up", "--link-root", "results"]));
        assert_eq!(std::fs::read_link(p.path("results/sims/Up/traj.tsv")).unwrap(), before);
        assert!(p.path("results/sims/up/traj.tsv").exists());
    }
}

/// A label longer than the store's rule allows is refused before anything is
/// stored.
#[test]
fn an_over_long_label_is_refused_before_anything_is_stored() {
    let p = Project::new();
    let label = "a".repeat(240);
    let err = p.refused(&sim(&["--seed", "1", "--label", &label, "--link-root", "results"]));
    assert!(err.contains("at most 64"), "{err}");
    assert_eq!(p.invocations(), 0, "nothing was stored");
}

/// A write refused part way, here by a directory camdl did not write at the
/// second cell's place, leaves no links and no manifest.
#[test]
fn a_write_refused_part_way_leaves_no_links() {
    let p = Project::new();
    p.ok(&sim(&["--seeds", "1:2", "--manifest", "build/pair.json"]));
    let ids: Vec<String> = results(&p.manifest("build/pair.json"))
        .iter()
        .filter(|r| r["kind"] == "sim")
        .map(|r| r["run_id"].as_str().unwrap()[..12].to_string())
        .collect();
    assert_eq!(ids.len(), 2);
    std::fs::create_dir_all(p.path(&format!("results/sims/{}", ids[1]))).unwrap();
    std::fs::write(p.path(&format!("results/sims/{}/mine.txt", ids[1])), "keep").unwrap();
    let err = p.refused(&sim(&["--seeds", "1:2", "--manifest", "build/again.json", "--link-root", "results"]));
    assert!(err.contains("mine.txt"), "{err}");
    assert!(links_under(&p.path("results")).is_empty(), "no cell's links were written");
    assert!(!p.path("build/again.json").exists(), "no manifest");
    assert_eq!(std::fs::read(p.path(&format!("results/sims/{}/mine.txt", ids[1]))).unwrap(), b"keep");
}

/// A group directory that is a symbolic link to another directory gets
/// links that resolve where they sit.
#[test]
fn links_in_a_symlinked_group_resolve() {
    let p = Project::new();
    std::fs::create_dir_all(p.path("results")).unwrap();
    std::fs::create_dir_all(p.path("elsewhere")).unwrap();
    // One level shallower than `results/sims`, so a target computed from
    // the link's apparent place would dangle.
    std::os::unix::fs::symlink("../elsewhere", p.path("results/sims")).unwrap();
    p.ok(&sim(&["--seed", "1", "--label", "x", "--manifest", "build/x.json", "--link-root", "results"]));
    // `links_under` does not follow the group's link, so read the link where it sits.
    let link = p.path("elsewhere/x/traj.tsv");
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    let stored = p.path_of(&["--manifest", "build/x.json", "traj"]);
    assert_eq!(link.canonicalize().unwrap(), stored.canonicalize().unwrap());
    assert_eq!(p.path("results/sims/x/traj.tsv").canonicalize().unwrap(), stored.canonicalize().unwrap());
}

/// A link root at or under the run store's root is refused, and nothing is
/// created there; a store inside the link root is allowed.
#[test]
fn a_link_root_in_the_store_is_refused() {
    let p = Project::new();
    p.ok(&sim(&["--seed", "1"]));
    for root in ["store", "store/links"] {
        let err = p.refused(&sim(&["--seed", "1", "--label", "x", "--link-root", root]));
        assert!(err.contains("inside the run store"), "{root}: {err}");
    }
    assert!(!p.path("store/links").exists() && !p.path("store/sims").exists());
    p.ok(&sim(&["--seed", "1", "--label", "x", "--link-root", "."]));
    assert!(p.path("sims/x/traj.tsv").exists(), "a store inside the link root is allowed");
}
