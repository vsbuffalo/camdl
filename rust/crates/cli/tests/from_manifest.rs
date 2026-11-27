//! `--from-manifest PATH`: `predict`, `simulate`, `profile` and `pfilter`
//! read the one fit run an output manifest lists (run spec §2.10), as a
//! workflow step reads the manifest an earlier step wrote.
//!
//! Drives the release binary on a three-compartment SIR model with a PGAS
//! fit, whose draws are a posterior, and an IF2 fit, whose `mle_params` is a
//! point estimate; each test has a working directory and run store of its
//! own.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build`", p.display());
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

/// A fit of beta and gamma, with the `[method]` body given.
fn fit_toml(method: &str) -> String {
    format!(
        "[model]\ncamdl = \"sir.ir.json\"\n\n[data.observations]\ncases = \"cases.tsv\"\n\n\
         [estimate.beta]\nbounds = [0.05, 2.0]\nstart  = 0.3\nprior  = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}\n\n\
         [estimate.gamma]\nbounds = [0.02, 0.9]\nstart  = 0.1\nprior  = {{ log_normal = {{ mu = -2.3, sigma = 0.5 }} }}\n\n\
         [fixed]\nN0    = 1000\n\n[method]\n{method}\n[config]\ndt = 1.0\n"
    )
}

const PGAS: &str = "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\nsweeps = 4\nburn_in = 1\nthin = 1\n";

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\niterations = 2\ncooling = 0.7\n";

/// A working directory holding the compiled model, its data and the two fit
/// configs, with the run store `store/` under it, the PGAS fit `@post`
/// (manifest `build/post.json`) and the IF2 fit `@mle` (`build/mle.json`).
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
        std::fs::write(dir.join("post.toml"), fit_toml(PGAS)).unwrap();
        std::fs::write(dir.join("mle.toml"), fit_toml(IF2)).unwrap();
        std::fs::create_dir(dir.join("build")).unwrap();
        let p = Project { tmp };
        p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "post", "--manifest", "build/post.json"]);
        p.ok(&["fit", "run", "mle.toml", "--seed", "1", "--label", "mle", "--manifest", "build/mle.json"]);
        p
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

    /// The refusal of `args`: exit 1 and its stderr.
    fn refused(&self, args: &[&str]) -> String {
        let out = self.run(args);
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert_eq!(out.status.code(), Some(1), "camdl {args:?} was not refused with exit 1:\n{err}");
        err
    }

    fn manifest(&self, rel: &str) -> serde_json::Value {
        let bytes = std::fs::read(self.path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        serde_json::from_slice(&bytes).unwrap()
    }

    fn write_manifest(&self, rel: &str, m: &serde_json::Value) {
        std::fs::write(self.path(rel), serde_json::to_vec_pretty(m).unwrap()).unwrap();
    }

    /// The run id of the one `fit_run` of the manifest at `rel`.
    fn fit_run_id(&self, rel: &str) -> String {
        let m = self.manifest(rel);
        let runs: Vec<&serde_json::Value> = results(&m).iter().filter(|r| r["kind"] == "fit_run").collect();
        assert_eq!(runs.len(), 1, "{rel} lists one fit run");
        runs[0]["run_id"].as_str().unwrap().to_string()
    }

    /// The outcome the newest invocation record gives each result of `kind`.
    fn last_outcomes(&self, kind: &str) -> Vec<String> {
        let dir = self.path("store/.invocations");
        let newest = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| e.file_name().to_str()?.strip_suffix(".json")?.parse::<u64>().ok())
            .max()
            .expect("an invocation record");
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{newest}.json"))).unwrap()).unwrap();
        record["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == kind)
            .map(|r| r["outcome"].as_str().unwrap().to_string())
            .collect()
    }
}

fn results(m: &serde_json::Value) -> &Vec<serde_json::Value> {
    m["results"].as_array().expect("results is an array")
}

/// The run ids of the results of `kind` the manifest at `rel` lists.
fn ids_of(p: &Project, rel: &str, kind: &str) -> Vec<String> {
    results(&p.manifest(rel)).iter().filter(|r| r["kind"] == kind).map(|r| r["run_id"].as_str().unwrap().to_string()).collect()
}

/// `args` and then `extra`.
fn with<'a>(args: &[&'a str], extra: &[&'a str]) -> Vec<&'a str> {
    args.iter().chain(extra).copied().collect()
}

/// Each command run once through a handle and once through
/// `--from-manifest`: the two store one result, by one run id, and the
/// second finds it stored.
#[test]
fn from_manifest_names_the_result_its_handle_names() {
    let p = Project::new();
    let post = p.fit_run_id("build/post.json");
    let mle = p.fit_run_id("build/mle.json");

    // predict: in place of the run handle.
    let predict = ["--mode", "free_forward", "--n-draws", "4"];
    p.ok(&with(&["predict", "@post"], &with(&predict, &["--manifest", "build/a.json"])));
    p.ok(&with(&["predict", "--from-manifest", "build/post.json"], &with(&predict, &["--manifest", "build/b.json"])));
    let a = ids_of(&p, "build/a.json", "prediction");
    assert_eq!(a.len(), 1);
    assert_eq!(a, ids_of(&p, "build/b.json", "prediction"), "predict: one prediction");
    assert_eq!(p.last_outcomes("prediction"), ["reused"], "predict: the second finds it stored");

    // simulate --draws posterior: in place of `--fit <run>`.
    let simulate = ["simulate", "sir.ir.json", "--draws", "posterior", "-n", "2", "--seed", "1"];
    p.ok(&with(&simulate, &["--fit", &post, "--manifest", "build/c.json"]));
    p.ok(&with(&simulate, &["--from-manifest", "build/post.json", "--manifest", "build/d.json"]));
    let c = ids_of(&p, "build/c.json", "sim");
    assert_eq!(c.len(), 2);
    assert_eq!(c, ids_of(&p, "build/d.json", "sim"), "simulate: the same cells");
    assert_eq!(p.last_outcomes("sim"), ["reused", "reused"], "simulate: the second finds them stored");

    // profile: the run's fit problem, as `--fit <run>` gives it.
    let profile = ["profile", "sir.ir.json", "--sweep", "beta=0.3,0.4", "--particles", "20", "--iterations", "1", "--n-starts", "1"];
    p.ok(&with(&profile, &["--fit", &mle, "--manifest", "build/e.json"]));
    p.ok(&with(&profile, &["--from-manifest", "build/mle.json", "--manifest", "build/f.json"]));
    let e = ids_of(&p, "build/e.json", "profile_point");
    assert_eq!(e.len(), 2);
    assert_eq!(e, ids_of(&p, "build/f.json", "profile_point"), "profile: the same points");
    assert_eq!(p.last_outcomes("profile_point"), ["reused", "reused"], "profile: the second finds them stored");

    // pfilter: the run's problem and its `mle_params` as the parameters.
    let mle_params = p.ok(&["path", &mle, "mle_params"]);
    let pfilter = ["pfilter", "sir.ir.json", "--particles", "20", "--seed", "1"];
    p.ok(&with(&pfilter, &["--fit", &mle, "--params", mle_params.trim(), "--manifest", "build/g.json"]));
    p.ok(&with(&pfilter, &["--from-manifest", "build/mle.json", "--manifest", "build/h.json"]));
    let g = ids_of(&p, "build/g.json", "pfilter");
    assert_eq!(g.len(), 1);
    assert_eq!(g, ids_of(&p, "build/h.json", "pfilter"), "pfilter: one filter run");
    assert_eq!(p.last_outcomes("pfilter"), ["reused"], "pfilter: the second finds it stored");
}

/// A manifest with no fit run, or with two, is refused, saying how many it
/// lists; so is a manifest of another schema version.
#[test]
fn from_manifest_reads_the_one_fit_run() {
    let p = Project::new();
    let mut none = p.manifest("build/post.json");
    none["results"].as_array_mut().unwrap().retain(|r| r["kind"] != "fit_run");
    p.write_manifest("build/none.json", &none);
    let err = p.refused(&["predict", "--from-manifest", "build/none.json", "--mode", "free_forward"]);
    assert!(err.contains("--from-manifest build/none.json lists no fit_run result, where one is needed"), "{err}");

    let mut two = p.manifest("build/post.json");
    let mle = p.manifest("build/mle.json");
    two["results"].as_array_mut().unwrap().extend(results(&mle).iter().filter(|r| r["kind"] == "fit_run").cloned());
    p.write_manifest("build/two.json", &two);
    let err = p.refused(&["predict", "--from-manifest", "build/two.json", "--mode", "free_forward"]);
    assert!(err.contains("--from-manifest build/two.json lists 2 fit_run results, where one is needed"), "{err}");

    for (schema, says) in [
        ("camdl.output-manifest/v1", "an output manifest of version `camdl.output-manifest/v1`; this camdl reads `camdl.output-manifest/v2`"),
        ("camdl.output-manifest/v3", "an output manifest of version `camdl.output-manifest/v3`; this camdl reads `camdl.output-manifest/v2`"),
        ("something/v1", "not an output manifest"),
    ] {
        let mut m = p.manifest("build/post.json");
        m["schema"] = schema.into();
        p.write_manifest("build/schema.json", &m);
        let err = p.refused(&["predict", "--from-manifest", "build/schema.json", "--mode", "free_forward"]);
        assert!(err.contains(&format!("--from-manifest build/schema.json: {says}")), "{schema}: {err}");
    }
}

/// The fit run must be in the store the command reads, and each of its files
/// must have the digest the manifest records.
#[test]
fn from_manifest_checks_the_store_against_the_manifest() {
    let p = Project::new();
    let post = p.fit_run_id("build/post.json");
    let out = p.run_in("other-store", &["predict", "--from-manifest", "build/post.json", "--mode", "free_forward"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("--from-manifest build/post.json") && err.contains(&post[..12]), "names the run id: {err}");

    let mut m = p.manifest("build/post.json");
    let run = m["results"].as_array_mut().unwrap().iter_mut().find(|r| r["kind"] == "fit_run").unwrap();
    let stored = run["files"]["draws"]["sha256"].as_str().unwrap().to_string();
    let other = "0".repeat(64);
    run["files"]["draws"]["sha256"] = other.clone().into();
    p.write_manifest("build/damaged.json", &m);
    let err = p.refused(&["predict", "--from-manifest", "build/damaged.json", "--mode", "free_forward"]);
    assert!(err.contains("does not describe the stored file `draws`"), "names the file: {err}");
    assert!(err.contains(&stored) && err.contains(&other), "names both digests: {err}");

    // A file the store holds and the manifest does not list is unchecked.
    let mut m = p.manifest("build/post.json");
    let run = m["results"].as_array_mut().unwrap().iter_mut().find(|r| r["kind"] == "fit_run").unwrap();
    run["files"].as_object_mut().unwrap().remove("draws");
    p.write_manifest("build/short.json", &m);
    let err = p.refused(&["predict", "--from-manifest", "build/short.json", "--mode", "free_forward"]);
    assert!(err.contains("does not describe the stored file `draws.tsv`") && err.contains("does not list it"), "{err}");
}

/// `pfilter --from-manifest` reads the run's `mle_params`, which a PGAS run
/// does not store.
#[test]
fn pfilter_from_manifest_needs_a_point_estimate() {
    let p = Project::new();
    let err = p.refused(&["pfilter", "sir.ir.json", "--particles", "20", "--from-manifest", "build/post.json"]);
    assert!(err.contains("mle_params") && err.contains("pgas"), "says what was found: {err}");
}

/// `--from-manifest` takes the place of the handle or `--fit`, and is refused
/// beside it.
#[test]
fn from_manifest_conflicts_with_the_form_it_replaces() {
    let p = Project::new();
    let post = p.fit_run_id("build/post.json");
    for args in [
        vec!["predict", "@post", "--from-manifest", "build/post.json"],
        vec!["simulate", "sir.ir.json", "--draws", "posterior", "--fit", &post, "--from-manifest", "build/post.json"],
        vec!["profile", "sir.ir.json", "--sweep", "beta=0.3", "--fit", &post, "--from-manifest", "build/post.json"],
        vec!["pfilter", "sir.ir.json", "--particles", "20", "--fit", &post, "--from-manifest", "build/post.json"],
    ] {
        let out = p.run(&args);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {err}");
        assert!(err.contains("cannot be used with"), "{args:?}: {err}");
    }
}
