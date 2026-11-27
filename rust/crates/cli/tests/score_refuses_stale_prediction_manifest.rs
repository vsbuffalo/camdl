//! `camdl score` labels each row with whether the parameters saw it, read
//! off the training rows a stored prediction's manifest records. A stored
//! `predictive.json` that records no `training_rows`, or carries another
//! schema tag than the one this build writes, is refused,
//! naming the file and saying to make the prediction again: read without its
//! training rows, a posterior prediction's rows would be labelled in sample
//! whether or not the fit held them out.
//!
//! Shells out to the built `camdl` binary.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", bin.display());
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
simulate { from = 0 'days  to = 20 'days }
"#;

const FIT_TOML: &str = r#"
[model]
camdl = "model.camdl"
[data]
file = "cases.tsv"
[estimate]
beta = { bounds = [0.1, 0.9], start = 0.4 }
[fixed]
gamma = 0.2
N0 = 1000
I0 = 5
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 2
cooling = 0.7
starts = "from_prior"
"#;

fn camdl_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("st"))
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn succeeds(out: Output, what: &str) -> Output {
    assert!(out.status.success(), "{what}: {}", String::from_utf8_lossy(&out.stderr));
    out
}

/// The folder of the one stored prediction under `dir/st`: the one holding
/// `predictive.json`.
fn prediction_folder(dir: &Path) -> PathBuf {
    let mut found = Vec::new();
    let mut stack = vec![dir.join("st")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "predictive.json") {
                found.push(p.parent().unwrap().to_path_buf());
            }
        }
    }
    assert_eq!(found.len(), 1, "one stored prediction: {found:?}");
    found.remove(0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Replace the stored `predictive.json` in `folder` by `bytes`, and the
/// digest its run record holds for it by theirs, so the store reads the
/// file as stored rather than refusing it as damaged.
fn store_manifest(folder: &Path, bytes: &[u8]) {
    let path = folder.join("predictive.json");
    let old = sha256_hex(&std::fs::read(&path).unwrap());
    let record = std::fs::read_to_string(folder.join("run.json")).unwrap();
    assert_eq!(record.matches(&old).count(), 1, "the run record holds the manifest's digest once");
    std::fs::write(folder.join("run.json"), record.replace(&old, &sha256_hex(bytes))).unwrap();
    std::fs::write(&path, bytes).unwrap();
}

#[test]
fn a_stored_prediction_whose_manifest_lacks_its_training_rows_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("model.toml"), FIT_TOML).unwrap();
    let cases = [3, 6, 9, 14, 20, 27, 33, 38, 40, 39, 36, 31, 26, 21, 16, 12, 9, 7, 5, 4];
    let mut tsv = String::from("time\tcases\n");
    for (t, c) in cases.iter().enumerate() {
        tsv.push_str(&format!("{}\t{c}\n", t + 1));
    }
    std::fs::write(dir.join("cases.tsv"), tsv).unwrap();
    succeeds(camdl_in(dir, &["fit", "run", "model.toml", "--seed", "1", "--label", "a", "--progress", "none"]), "fit run");
    succeeds(camdl_in(dir, &["predict", "@a", "--label", "p", "--mode", "free_forward", "--progress", "none"]), "predict");
    let folder = prediction_folder(dir);
    let original = std::fs::read(folder.join("predictive.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();

    // The manifest an earlier build wrote: the training rows under
    // `parameter_set`, and the tag it was written under.
    let mut renamed = manifest.clone();
    let rows = renamed.as_object_mut().unwrap().remove("training_rows").unwrap();
    renamed["parameter_set"] = rows;
    let mut old_tag = manifest.clone();
    old_tag["schema"] = "camdl.predictive/v5".into();
    let cases: [(&str, Vec<u8>, &str); 2] = [
        ("no training_rows", serde_json::to_vec_pretty(&renamed).unwrap(), "training_rows"),
        ("an earlier schema tag", serde_json::to_vec_pretty(&old_tag).unwrap(), "camdl.predictive/v5"),
    ];
    for (what, bytes, named) in cases {
        store_manifest(&folder, &bytes);
        let out = camdl_in(dir, &["score", "@p", "--data", "cases=cases.tsv", "--progress", "none"]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{what}: the score is refused:\n{err}");
        assert!(err.contains("predictive.json"), "{what}: the refusal names the file:\n{err}");
        assert!(err.contains(named), "{what}: the refusal says `{named}`:\n{err}");
        assert!(err.contains("make the prediction again"), "{what}: the refusal says to make the prediction again:\n{err}");
        assert!(!err.contains("internal error"), "{what}: an earlier build's manifest is not an internal error:\n{err}");
    }

    store_manifest(&folder, &original);
    succeeds(camdl_in(dir, &["score", "@p", "--data", "cases=cases.tsv", "--progress", "none"]), "score of the manifest as written");
}
