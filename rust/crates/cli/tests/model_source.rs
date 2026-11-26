//! A command that compiles a model stores its model source: the file, the
//! documentation, and for a `.camdl` the rendered mathematics and flow graph,
//! as `camdl render` writes them, depending on the quantities the compile
//! produced. The compiled model's listing in the command's invocation record
//! names that source. A model given as compiled IR has no render, and the
//! command says so. A command that stores a result from a stored one records
//! the source too, so a reader finds it from either result.
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
#' A closed SIR for the model source test.
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

/// Every stored model source folder under `store`.
fn sources(store: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.parent().and_then(|p| p.file_name()).is_some_and(|n| n == "model_source") {
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

    let found = sources(&store);
    assert_eq!(found.len(), 1, "one model source: {found:?}");
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
    assert_eq!(stderr.matches("is compiled IR, so its stored model source has no rendered mathematics").count(), 1, "{stderr}");
    let found = sources(&store);
    assert_eq!(found.len(), 1, "one model source: {found:?}");
    assert!(found[0].join("source.ir.json").exists());
    assert!(!found[0].join("model.render.json").exists());
    assert!(!found[0].join("model.graph.json").exists());
}

/// A reader that names the simulation by its folder, as camdl-scope does,
/// is given the model source of the model the simulation ran.
#[test]
fn a_result_folder_names_its_model_source() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let store = tmp.path().join("store");
    simulate(&model, &store);
    let source = &sources(&store)[0];

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
    let out = camdl(&["path", &cell.to_string_lossy(), "model_source:render", "--store", &store.to_string_lossy()]);
    assert!(out.status.success(), "path: {}", String::from_utf8_lossy(&out.stderr));
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        std::fs::canonicalize(&printed).unwrap(),
        std::fs::canonicalize(source.join("model.render.json")).unwrap()
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

/// The documentation of `beta` that `camdl cat HANDLE model_source:docs`
/// prints.
fn beta_doc(dir: &Path, handle: &str) -> serde_json::Value {
    let out = succeeds(camdl_in(dir, &["cat", handle, "model_source:docs"]), &format!("cat {handle}"));
    let docs: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    docs["parameters"]["beta"]["text"].clone()
}

/// A prediction made from a stored fit run reads the run's model: the
/// model's documentation is found from the prediction by label, by folder
/// and by id, and it is the documentation of the file the fit was run from.
#[test]
fn a_prediction_from_a_stored_fit_names_the_fits_model_source() {
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

/// A run whose invocation record names no model source for its model is
/// summarised without the model's documentation, and the summary says so
/// once.
#[test]
fn a_summary_with_no_recorded_model_file_says_so() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    succeeds(camdl_in(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--progress", "none"]), "fit run");
    let record = dir.join("st/.invocations/1.json");
    let mut inv: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
    for r in inv["results"].as_array_mut().unwrap() {
        r.as_object_mut().unwrap().remove("source");
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
    let first = folders_under(&dir.join("st"), "model_source")
        .into_iter()
        .find(|p| std::fs::read_to_string(p.join("docs.json")).unwrap().contains("FIRST"))
        .expect("the first file's model source");
    let first = first.to_string_lossy().rsplit('-').next().unwrap().to_string();

    let notice = format!(
        "info: {id}: the model's documentation is the one of the model file the latest invocation that listed it \
         (invocation 2) ran; earlier invocations that listed the result ran other files (model sources {first} (@a)), \
         which their labels show"
    );
    let out = succeeds(camdl_in(dir, &["fit", "summary", &id, "--parameters"]), "fit summary");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(stderr.matches(&notice).count(), 1, "{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("SECOND doc of beta."));
    let out = succeeds(camdl_in(dir, &["cat", &id, "model_source:docs"]), "cat");
    assert_eq!(String::from_utf8_lossy(&out.stderr).matches(&notice).count(), 1, "{}", String::from_utf8_lossy(&out.stderr));

    for (label, doc) in [("@a", "FIRST doc of beta."), ("@b", "SECOND doc of beta.")] {
        let out = succeeds(camdl_in(dir, &["fit", "summary", label, "--parameters"]), "fit summary");
        assert!(String::from_utf8_lossy(&out.stdout).contains(doc), "{label}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("model-source"), "{label}");
        assert!(!String::from_utf8_lossy(&out.stderr).contains("earlier invocations"), "{label}");
    }
}

/// The compiled model's listing in a command's record names its model
/// source by kind and run id, and its attributes hold only its name; the
/// source depends on the quantities the compile produced and holds the
/// file's name.
#[test]
fn a_compiled_models_listing_names_its_model_source() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    succeeds(camdl_in(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "f", "--progress", "none"]), "fit run");
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("st/.invocations/1.json")).unwrap()).unwrap();
    let models: Vec<&serde_json::Value> =
        record["results"].as_array().unwrap().iter().filter(|r| r["kind"] == "compiled_model").collect();
    assert!(!models.is_empty(), "{record:#}");
    for m in &models {
        assert_eq!(m["attributes"], serde_json::json!({ "name": "model" }), "{record:#}");
        assert_eq!(m["source"]["kind"], "model_source", "{record:#}");
    }
    let source_id = models[0]["source"]["run_id"].as_str().unwrap();
    let folder = folders_under(&dir.join("st"), "model_source")
        .into_iter()
        .find(|p| p.to_string_lossy().ends_with(&source_id[..12]))
        .expect("the listed model source is stored");
    let identity: serde_json::Value = serde_json::from_slice(&std::fs::read(folder.join("identity.json")).unwrap()).unwrap();
    assert_eq!(identity["identity"]["file_name"], "model.camdl", "{identity:#}");
    let run: serde_json::Value = serde_json::from_slice(&std::fs::read(folder.join("run.json")).unwrap()).unwrap();
    let deps: Vec<&str> = run["deps"].as_array().unwrap().iter().map(|d| d["kind"].as_str().unwrap()).collect();
    assert_eq!(deps, ["quantities"], "the source depends on its quantities: {run:#}");
    let others: Vec<&serde_json::Value> =
        record["results"].as_array().unwrap().iter().filter(|r| r["kind"] != "compiled_model").collect();
    assert!(others.iter().all(|r| r.get("source").is_none()), "only a compiled model's listing names a source: {record:#}");
}

/// A fit that runs until it is killed: [`FIT_TOML`] with many iterations of
/// many particles.
const SLOW_FIT_TOML: &str = r#"
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
particles = 20000
iterations = 100000
cooling = 0.7
starts = "from_prior"
"#;

/// A fit process, killed when dropped.
struct Running(std::process::Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The folder of the one fit run under `dir/st` once its `run.json` says it
/// is running, waiting up to two minutes for it.
fn running_run_folder(dir: &Path, fit: &mut Running) -> PathBuf {
    let start = std::time::Instant::now();
    loop {
        assert!(fit.0.try_wait().unwrap().is_none(), "the slow fit ended before it was read");
        let runs: Vec<PathBuf> = if dir.join("st").is_dir() {
            folders_under(&dir.join("st"), "runs")
                .into_iter()
                .flat_map(|m| std::fs::read_dir(m).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()))
                .collect()
        } else {
            Vec::new()
        };
        let running = runs.iter().find(|r| {
            std::fs::read(r.join("run.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .is_some_and(|v| v["status"] == "running")
        });
        if let Some(r) = running {
            return r.clone();
        }
        assert!(start.elapsed() < std::time::Duration::from_secs(120), "no running fit run under {}", dir.display());
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// A fit run that has not completed is read for its model source: the
/// invocation running it answers for it, and the reader says that
/// invocation has not ended. The run's own files are refused until it
/// completes.
#[test]
fn a_running_fit_names_its_model_source_and_refuses_its_own_files() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    std::fs::write(dir.join("slow.toml"), SLOW_FIT_TOML).unwrap();
    let child = Command::new(binary())
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("st"))
        .args(["fit", "run", "slow.toml", "--seed", "1", "--progress", "none"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn camdl fit run");
    let mut fit = Running(child);
    let run = running_run_folder(dir, &mut fit);
    let handle = run.to_string_lossy().into_owned();

    let out = camdl_in(dir, &["path", &handle, "model_source:docs"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "path of a running fit's model source: {stderr}");
    let printed = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let source = folders_under(&dir.join("st"), "model_source");
    assert_eq!(source.len(), 1, "{source:?}");
    assert_eq!(
        std::fs::canonicalize(&printed).unwrap(),
        std::fs::canonicalize(source[0].join("docs.json")).unwrap(),
        "{stderr}"
    );
    assert!(stderr.contains("which has not ended"), "the reader says the answering invocation is running: {stderr}");

    let cat = camdl_in(dir, &["cat", &handle, "model_source:docs"]);
    assert!(cat.status.success(), "cat of a running fit's model source: {}", String::from_utf8_lossy(&cat.stderr));
    let docs: serde_json::Value = serde_json::from_slice(&cat.stdout).unwrap();
    assert_eq!(docs["parameters"]["beta"]["text"], "The transmission rate, per day.");

    for own in ["fit_meta", "summary"] {
        let out = camdl_in(dir, &["path", &handle, own]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "`path {own}` of a running fit is refused: {stderr}");
        assert!(stderr.contains("did not complete and is not read") && stderr.contains("is still running"), "{own}: {stderr}");
    }
}

/// A fit run that failed is not read for its model source: its invocation's
/// listing of it failed, so no invocation answers for it, and the lookup is
/// refused as a lookup of its own files is, with no warning about a missing
/// source.
#[test]
fn a_failed_fit_is_refused_for_its_model_source() {
    let tmp = fit_inputs();
    let dir = tmp.path();
    succeeds(camdl_in(dir, &["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"]), "fit run");
    let run = run_folder(dir);
    let failure = serde_json::json!({ "operation": "if2", "reason": "disk full" });
    let run_json = run.join("run.json");
    let mut rec: serde_json::Value = serde_json::from_slice(&std::fs::read(&run_json).unwrap()).unwrap();
    rec["status"] = "failed".into();
    rec["failure"] = failure.clone();
    std::fs::write(&run_json, serde_json::to_vec_pretty(&rec).unwrap()).unwrap();
    let inv_path = dir.join("st/.invocations/1.json");
    let mut inv: serde_json::Value = serde_json::from_slice(&std::fs::read(&inv_path).unwrap()).unwrap();
    for r in inv["results"].as_array_mut().unwrap() {
        if r["kind"] == "fit_run" {
            r["outcome"] = "failed".into();
            r["failure"] = failure.clone();
        }
    }
    std::fs::write(&inv_path, serde_json::to_vec_pretty(&inv).unwrap()).unwrap();

    let out = camdl_in(dir, &["path", &run.to_string_lossy(), "model_source:docs"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a failed fit's model source is refused: {stderr}");
    assert!(stderr.contains("did not complete and is not read") && stderr.contains("failed while if2: disk full"), "{stderr}");
    assert!(!stderr.contains("no invocation recorded"), "{stderr}");
}
