//! A command that compiles a model stores the model's presentation: the
//! source, the documentation, and for a `.camdl` the rendered mathematics and
//! flow graph, as `camdl render` writes them. A model given as compiled IR
//! has no render, and the command says so. A command that stores a result
//! from a stored one records the model's presentation too, so a reader finds
//! it from either result.
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
#' A closed SIR for the presentation test.
time_unit = 'days
compartments { S, I, R }
parameters {
  #' The transmission rate.
  beta  : rate in [0.01, 2.0]
  gamma : rate in [0.01, 1.0]
}
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 5 'days }
scenarios { baseline { set = { beta = 0.3  gamma = 0.1 } } }
"#;

fn camdl(args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn simulate(model: &Path, store: &Path) -> Output {
    let out = camdl(&[
        "simulate",
        &model.to_string_lossy(),
        "--scenario",
        "baseline",
        "--seed",
        "1",
        "--store",
        &store.to_string_lossy(),
        "-o",
        &store.with_extension("tsv").to_string_lossy(),
    ]);
    assert!(out.status.success(), "simulate: {}", String::from_utf8_lossy(&out.stderr));
    out
}

/// Every stored presentation folder under `store`.
fn presentations(store: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.parent().and_then(|p| p.file_name()).is_some_and(|n| n == "model_presentation") {
                found.push(path);
            } else {
                stack.push(path);
            }
        }
    }
    found
}

#[test]
fn a_camdl_model_stores_its_render_and_graph() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let store = tmp.path().join("store");
    simulate(&model, &store);

    let found = presentations(&store);
    assert_eq!(found.len(), 1, "one presentation: {found:?}");
    let dir = &found[0];
    for (file, format) in [("model.render.json", "json"), ("model.graph.json", "graph")] {
        let rendered = camdl(&["render", &model.to_string_lossy(), "--format", format]);
        assert!(rendered.status.success());
        assert_eq!(std::fs::read(dir.join(file)).unwrap(), rendered.stdout, "{file} is what `camdl render` writes");
    }
    assert_eq!(std::fs::read_to_string(dir.join("source.camdl")).unwrap(), MODEL);
    let docs: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("docs.json")).unwrap()).unwrap();
    assert_eq!(docs["parameters"]["beta"]["text"], "The transmission rate.");
}

#[test]
fn a_compiled_model_has_no_render_and_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let ir = tmp.path().join("sir.ir.json");
    let camdlc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} - run `make build-ocaml`", camdlc.display());
    let compiled = Command::new(camdlc).env_remove("CLICOLOR_FORCE").arg(&model).arg("-o").arg(&ir).output().unwrap();
    assert!(compiled.status.success(), "compile: {}", String::from_utf8_lossy(&compiled.stderr));
    let store = tmp.path().join("store");
    let out = simulate(&ir, &store);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(stderr.matches("is compiled IR, so its stored presentation has no rendered mathematics").count(), 1, "{stderr}");
    let found = presentations(&store);
    assert_eq!(found.len(), 1, "one presentation: {found:?}");
    assert!(found[0].join("source.ir.json").exists());
    assert!(!found[0].join("model.render.json").exists());
    assert!(!found[0].join("model.graph.json").exists());
}

/// A reader that names the simulation by its folder, as camdl-scope does,
/// is given the presentation of the model the simulation ran.
#[test]
fn a_result_folder_names_its_model_presentation() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let store = tmp.path().join("store");
    simulate(&model, &store);
    let presentation = &presentations(&store)[0];

    let mut stack = vec![store.clone()];
    let mut cell = None;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.join("traj.tsv").exists() {
                    cell = Some(path.clone());
                }
                stack.push(path);
            }
        }
    }
    let cell = cell.expect("a simulated cell");
    let out = camdl(&["path", &cell.to_string_lossy(), "model_presentation:render", "--store", &store.to_string_lossy()]);
    assert!(out.status.success(), "path: {}", String::from_utf8_lossy(&out.stderr));
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        std::fs::canonicalize(&printed).unwrap(),
        std::fs::canonicalize(presentation.join("model.render.json")).unwrap()
    );
    let traj = camdl(&["path", &cell.to_string_lossy(), "traj", "--store", &store.to_string_lossy()]);
    assert!(traj.status.success(), "the cell's own files are found as before: {}", String::from_utf8_lossy(&traj.stderr));
}

const FIT_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  #' The transmission rate, per day.
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

/// A directory holding `model.camdl`, `fit.toml` and `cases.tsv`, for a
/// small IF2 fit whose model documents `beta`.
fn fit_inputs() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), FIT_MODEL).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), FIT_TOML).unwrap();
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

/// Every folder under `root` whose parent folder is named `parent`.
fn folders_under(root: &Path, parent: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.parent().and_then(|p| p.file_name()).is_some_and(|n| n == parent) {
                    found.push(path.clone());
                }
                stack.push(path);
            }
        }
    }
    found
}

/// The documentation of `beta` that `camdl cat HANDLE model_presentation:docs`
/// prints.
fn beta_doc(dir: &Path, handle: &str) -> serde_json::Value {
    let out = succeeds(camdl_in(dir, &["cat", handle, "model_presentation:docs"]), &format!("cat {handle}"));
    let docs: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    docs["parameters"]["beta"]["text"].clone()
}

/// A prediction made from a stored fit run reads the run's model: the
/// model's documentation is found from the prediction by label, by folder
/// and by id, and it is the documentation of the file the fit was run from.
#[test]
fn a_prediction_from_a_stored_fit_names_the_fits_presentation() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    succeeds(camdl_in(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "a", "--progress", "none"]), "fit run");
    succeeds(camdl_in(dir, &["predict", "@a", "--label", "p", "--progress", "none"]), "predict");

    let predictions = folders_under(&dir.join("st"), "predict");
    assert_eq!(predictions.len(), 1, "one prediction: {predictions:?}");
    let folder = predictions[0].to_string_lossy().into_owned();
    let id = folder.rsplit('-').next().unwrap().to_string();
    for handle in ["@p", folder.as_str(), id.as_str()] {
        assert_eq!(beta_doc(dir, handle), "The transmission rate, per day.", "{handle}");
    }

    // A score of that prediction reads the same model.
    succeeds(camdl_in(dir, &["score", "@p", "--data", "cases=cases.tsv", "--label", "s", "--progress", "none"]), "score");
    assert_eq!(beta_doc(dir, "@s"), "The transmission rate, per day.", "@s");
}

/// The one fit run's folder under `dir/st`.
fn run_folder(dir: &Path) -> PathBuf {
    let runs: Vec<PathBuf> = folders_under(&dir.join("st"), "runs")
        .into_iter()
        .flat_map(|method| std::fs::read_dir(method).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()))
        .collect();
    assert_eq!(runs.len(), 1, "one fit run: {runs:?}");
    runs[0].clone()
}

/// A run whose invocation record names no presentation for its model, as a
/// record written before records named one does, is summarised without
/// the model's documentation, and the summary says so once.
#[test]
fn a_summary_with_no_recorded_model_file_says_so() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    succeeds(camdl_in(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--progress", "none"]), "fit run");
    let record = dir.join("st/.invocations/1.json");
    let mut inv: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    for r in inv["results"].as_array_mut().unwrap() {
        r["attributes"].as_object_mut().unwrap().remove("presentation");
    }
    std::fs::write(&record, serde_json::to_vec_pretty(&inv).unwrap()).unwrap();

    let out = succeeds(camdl_in(dir, &["fit", "summary", "@f", "--parameters"]), "fit summary");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr.matches("warning: @f: the model's documentation is not shown: the invocation @f names (invocation 1) recorded no model file").count(),
        1,
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&out.stdout).contains("The transmission rate, per day."));
}

/// Two model files that differ only in a doc line, fitted to one run: the
/// run's id shows the latest invocation's file and names the other, with
/// its label; each label shows its own file and says nothing.
#[test]
fn an_id_whose_invocations_ran_different_files_names_the_others() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    for (label, doc) in [("a", "FIRST doc of beta."), ("b", "SECOND doc of beta.")] {
        std::fs::write(dir.join(format!("{label}.camdl")), FIT_MODEL.replace("The transmission rate, per day.", doc)).unwrap();
        std::fs::write(dir.join(format!("{label}.toml")), FIT_TOML.replace("model.camdl", &format!("{label}.camdl"))).unwrap();
        let toml = format!("{label}.toml");
        succeeds(camdl_in(dir, &["fit", "run", &toml, "--seed", "1", "--label", label, "--progress", "none"]), "fit run");
    }
    let folder = run_folder(dir);
    let id = folder.to_string_lossy().rsplit('-').next().unwrap().to_string();
    let first = folders_under(&dir.join("st"), "model_presentation")
        .into_iter()
        .find(|p| std::fs::read_to_string(p.join("docs.json")).unwrap().contains("FIRST"))
        .expect("the first file's presentation");
    let first = first.to_string_lossy().rsplit('-').next().unwrap().to_string();

    let notice = format!(
        "info: {id}: the model's documentation is the one of the model file the latest invocation that listed it \
         (invocation 2) ran; earlier invocations that listed the result ran other files (presentations {first} (@a)), \
         which their labels show"
    );
    let out = succeeds(camdl_in(dir, &["fit", "summary", &id, "--parameters"]), "fit summary");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(stderr.matches(&notice).count(), 1, "{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("SECOND doc of beta."));
    let out = succeeds(camdl_in(dir, &["cat", &id, "model_presentation:docs"]), "cat");
    assert_eq!(String::from_utf8_lossy(&out.stderr).matches(&notice).count(), 1, "{}", String::from_utf8_lossy(&out.stderr));

    for (label, doc) in [("@a", "FIRST doc of beta."), ("@b", "SECOND doc of beta.")] {
        let out = succeeds(camdl_in(dir, &["fit", "summary", label, "--parameters"]), "fit summary");
        assert!(String::from_utf8_lossy(&out.stdout).contains(doc), "{label}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("model-presentation"), "{label}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("earlier invocations"), "{label}");
    }
}
