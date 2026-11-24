//! A command that compiles a model stores the model's presentation: the
//! source, the documentation, and for a `.camdl` the rendered mathematics and
//! flow graph, as `camdl render` writes them. A model given as compiled IR
//! has no render, and the command says so.
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
