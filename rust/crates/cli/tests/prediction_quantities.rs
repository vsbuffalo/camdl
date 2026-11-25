//! The quantities a prediction from a stored fit evaluates are those of the
//! model source the answering invocation recorded: the file `fit run` was
//! given, or the one `predict --model FILE` gives. No other invocation's
//! quantities stand in. A prediction is scored with its own quantities, the
//! ones its identity holds.
//!
//! Shells out to the built `camdl` binary.

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

/// The model with the quantities block `quantities`.
fn with_quantities(quantities: &str) -> String {
    format!("{MODEL}\nquantities {{\n  {quantities}\n}}\n")
}

/// A directory holding `cases.tsv` and, for each `(name, quantity)`,
/// `name.camdl` declaring that quantity and `name.toml` fitting it.
fn inputs(files: &[(&str, &str)]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for (name, quantity) in files {
        std::fs::write(tmp.path().join(format!("{name}.camdl")), with_quantities(quantity)).unwrap();
        std::fs::write(tmp.path().join(format!("{name}.toml")), FIT_TOML.replace("model.camdl", &format!("{name}.camdl"))).unwrap();
    }
    let cases = [3, 6, 9, 14, 20, 27, 33, 38, 40, 39, 36, 31, 26, 21, 16, 12, 9, 7, 5, 4];
    let mut tsv = String::from("time\tcases\n");
    for (t, c) in cases.iter().enumerate() {
        tsv.push_str(&format!("{}\t{c}\n", t + 1));
    }
    std::fs::write(tmp.path().join("cases.tsv"), tsv).unwrap();
    tmp
}

/// `camdl ARGS` run in `dir` against the store `dir/st`.
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

fn fit(dir: &Path, name: &str, label: &str) {
    let toml = format!("{name}.toml");
    succeeds(camdl_in(dir, &["fit", "run", &toml, "--seed", "1", "--label", label, "--progress", "none"]), "fit run");
}

/// The one fit run's id, from its folder under `dir/st`.
fn run_id(dir: &Path) -> String {
    let mut found = Vec::new();
    let mut stack = vec![dir.join("st")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.parent().and_then(|q| q.parent()).and_then(|q| q.file_name()).is_some_and(|n| n == "runs") {
                    found.push(p.clone());
                }
                stack.push(p);
            }
        }
    }
    assert_eq!(found.len(), 1, "one fit run: {found:?}");
    found[0].to_string_lossy().rsplit('-').next().unwrap().to_string()
}

const PREDICT: &[&str] = &["--mode", "free_forward", "--progress", "none"];

/// A prediction made through `--model FILE` holds FILE's quantities, and is
/// scored with them: the score poses the prediction again with its own
/// quantities, not the run's recorded ones.
#[test]
fn a_prediction_made_with_another_model_file_scores() {
    let tmp = inputs(&[("model", "peak_I = max(I)"), ("edited", "peak_R = max(R)")]);
    let dir = tmp.path();
    fit(dir, "model", "a");
    let mut args = vec!["predict", "@a", "--model", "edited.camdl", "--label", "p"];
    args.extend_from_slice(PREDICT);
    succeeds(camdl_in(dir, &args), "predict --model");
    succeeds(camdl_in(dir, &["score", "@p", "--data", "cases=cases.tsv", "--label", "s", "--progress", "none"]), "score");
}

/// A run whose answering invocation recorded no model source has no known
/// quantities: `predict` refuses and says how to give them, and no earlier
/// record stands in.
#[test]
fn a_prediction_of_a_run_with_no_recorded_source_is_refused() {
    let tmp = inputs(&[("model", "peak_I = max(I)")]);
    let dir = tmp.path();
    fit(dir, "model", "f");
    let record = dir.join("st/.invocations/1.json");
    let mut inv: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    for r in inv["results"].as_array_mut().unwrap() {
        r.as_object_mut().unwrap().remove("source");
    }
    std::fs::write(&record, serde_json::to_vec_pretty(&inv).unwrap()).unwrap();
    let mut args = vec!["predict", "@f"];
    args.extend_from_slice(PREDICT);
    let out = camdl_in(dir, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "no source is recorded, so no quantities are known: {stderr}");
    assert!(stderr.contains("no invocation recorded the model file") && stderr.contains("--model FILE"), "{stderr}");
}

/// Two files that differ only in their quantities, fitted to one run: the run
/// id names both, so `predict <run id>` refuses and asks for a label, and
/// each label predicts with its own file's quantities.
#[test]
fn a_run_id_whose_sources_differ_in_quantities_is_refused() {
    let tmp = inputs(&[("a", "peak_I = max(I)"), ("b", "peak_R = max(R)")]);
    let dir = tmp.path();
    fit(dir, "a", "a");
    fit(dir, "b", "b");
    let id = run_id(dir);
    let mut args = vec!["predict", id.as_str()];
    args.extend_from_slice(PREDICT);
    let out = camdl_in(dir, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("different quantities") && stderr.contains("@a") && stderr.contains("@label"), "{stderr}");
    for label in ["@a", "@b"] {
        let mut args = vec!["predict", label];
        args.extend_from_slice(PREDICT);
        succeeds(camdl_in(dir, &args), label);
    }
}

/// A label names one invocation, which answers alone, for the model's name
/// too: two files that differ only in a comment pose one problem, and
/// `predict @a` names the model as `@a`'s file does, not as the later one.
#[test]
fn a_label_names_the_model_as_its_own_invocation_did() {
    let tmp = inputs(&[("a", "peak_I = max(I)")]);
    let dir = tmp.path();
    let a = std::fs::read_to_string(dir.join("a.camdl")).unwrap();
    std::fs::write(dir.join("c.camdl"), format!("# the same model, commented\n{a}")).unwrap();
    std::fs::write(dir.join("c.toml"), FIT_TOML.replace("model.camdl", "c.camdl")).unwrap();
    fit(dir, "a", "a");
    fit(dir, "c", "c");
    let mut args = vec!["predict", "@a", "--label", "p"];
    args.extend_from_slice(PREDICT);
    succeeds(camdl_in(dir, &args), "predict @a");
    let records = dir.join("st/.invocations");
    let last = std::fs::read_dir(&records)
        .unwrap()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".json")?.parse::<u64>().ok())
        .max()
        .unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(records.join(format!("{last}.json"))).unwrap()).unwrap();
    let names: Vec<&serde_json::Value> = record["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "compiled_model" || r["kind"] == "prediction")
        .map(|r| r["attributes"].get("name").or(r["attributes"].get("model")).unwrap())
        .collect();
    assert!(!names.is_empty() && names.iter().all(|n| *n == "a"), "the model is named as @a's file: {record:#}");
}
