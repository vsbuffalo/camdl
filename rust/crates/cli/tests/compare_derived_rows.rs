//! What `camdl compare` does around a derived row: the filter runs it shells
//! out to, and what it checks about the inputs it re-reads.
//!
//! Every test fits two small IF2 fits of the committed SEIR golden (no
//! `camdlc` needed) and compares them by `fit.toml` path, so the
//! prequential traces are derived by `camdl pfilter` children.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

fn golden_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/seir_observations.ir.json")
        .canonicalize()
        .unwrap()
}

const OBS: &str = "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n";

/// Write and run one IF2 fit in `dir/name`, training on `dir/name/obs.tsv`.
fn fit(dir: &Path, name: &str, beta: f64) -> PathBuf {
    let sub = dir.join(name);
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("obs.tsv"), OBS).unwrap();
    std::fs::write(sub.join("fit.toml"), format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "obs.tsv"

[estimate.I0]
bounds = [1, 1000]
start  = 5

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = {beta}

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().display())).unwrap();
    let out = Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", "fit.toml", "--seed", "1"])
        .current_dir(&sub)
        .output().expect("spawn camdl fit run");
    assert!(out.status.success(), "fit run in {} failed:\n{}",
        sub.display(), String::from_utf8_lossy(&out.stderr));
    sub
}

/// `camdl compare a/fit.toml b/fit.toml` run from `dir`, with no output
/// root in the environment: a child that recorded a leaf would put it under
/// `dir/results`.
fn compare(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["compare", "a/fit.toml", "b/fit.toml",
        "--particles", "100", "--draws", "1", "--replicates", "2"];
    args.extend_from_slice(extra);
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(&args)
        .current_dir(dir)
        .output().expect("spawn camdl compare")
}

fn find_fit_meta(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                return p;
            }
        }
    }
    panic!("no fit.meta.json under {}", root.display());
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// `compare` re-reads each fit's training file from its path. A file edited
/// in place after the fit ran used to be scored at a θ̂ fitted to other
/// bytes with nothing said: the shared-data check reads only the hashes
/// recorded at fit time, and they still agree. Now it is a refusal (exit 2)
/// naming the row, the stream, the path and both hashes; with
/// `--allow-changed-data` the comparison renders and every format says so.
#[test]
fn a_training_file_edited_after_the_fit_is_refused_and_overridable() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fit_a = fit(dir, "a", 0.10);
    fit(dir, "b", 0.15);

    // Unedited: no statement either way.
    let out = compare(dir, &["--format", "json"]);
    assert!(out.status.success(), "compare failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert!(v["data_changed"].is_null(), "nothing changed:\n{v}");
    assert!(v["data_change_unchecked"].is_null(), "every file was checked:\n{v}");

    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(find_fit_meta(&fit_a.join("results"))).unwrap()).unwrap();
    let recorded = meta["data_hashes"]["weekly_cases"].as_str()
        .expect("the fit recorded a hash for weekly_cases").to_string();
    let edited = OBS.replace("35\t5", "35\t9");
    std::fs::write(fit_a.join("obs.tsv"), &edited).unwrap();
    let current = sha256_hex(edited.as_bytes());

    let out = compare(dir, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2),
        "an edited training file is a commensurability refusal:\n{stderr}");
    for needle in ["changed after the fit ran", "'a/fit.toml'", "'weekly_cases'",
                   "obs.tsv", &recorded[..8], &current[..8], "--allow-changed-data"] {
        assert!(stderr.contains(needle), "the refusal names {needle:?}:\n{stderr}");
    }
    assert!(!stderr.contains("'b/fit.toml' stream"),
        "the unedited fit is not named:\n{stderr}");

    let out = compare(dir, &["--allow-changed-data"]);
    let table = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the override renders:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(table.contains("changed after the fit ran") && table.contains(&current[..8]),
        "the table states the mismatch:\n{table}");

    let out = compare(dir, &["--allow-changed-data", "--format", "json"]);
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    assert!(v["data_changed"].as_str().is_some_and(|s|
            s.contains("weekly_cases") && s.contains(&recorded[..8]) && s.contains(&current[..8])),
        "the JSON states the mismatch as a field:\n{v}");
}

/// `compare` derives each row by running `camdl pfilter` once per draw per
/// replicate. Each of those runs used to record a `pfilters/` leaf under the
/// caller's output root — a log-likelihood nobody asked for, written into
/// whatever directory `compare` was run from.
#[test]
fn deriving_a_trace_writes_no_pfilter_leaf() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, "a", 0.10);
    fit(dir, "b", 0.15);

    let out = compare(dir, &[]);
    assert!(out.status.success(), "compare failed:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(!dir.join("results").join("pfilters").exists(),
        "compare's filter runs must not record pfilters/ leaves under {}",
        dir.join("results").display());
}
