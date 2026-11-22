//! Finding what `camdl fit run` stored in a run store, for the tests that run
//! it. A store's result folders each hold an `identity.json` naming their
//! kind: a fit problem is `fit` (`<build>/fit/<model>-<id12>/`), a fit run is
//! `fit_run` (`runs/<method>-<id12>/seed_<n>-<id12>/` inside its problem), and
//! a prediction is `prediction` (`predict/<mode>-<id12>/` inside its run).

#![allow(dead_code)] // each test file uses part of it

use std::path::{Path, PathBuf};

/// Every result folder of `kind` under the store at `root`, in path order.
pub fn folders_of(root: &Path, kind: &str) -> Vec<PathBuf> {
    fn walk(dir: &Path, kind: &str, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let ident = p.join("identity.json");
            if ident.is_file() {
                let id: serde_json::Value = serde_json::from_slice(&std::fs::read(&ident).unwrap()).unwrap();
                if id["kind"] == kind {
                    out.push(p.clone());
                }
            }
            walk(&p, kind, out);
        }
    }
    let mut out = Vec::new();
    walk(root, kind, &mut out);
    out
}

/// Every stored fit run's folder under `root`.
pub fn runs(root: &Path) -> Vec<PathBuf> {
    folders_of(root, "fit_run")
}

/// Every stored fit problem's folder under `root`.
pub fn problems(root: &Path) -> Vec<PathBuf> {
    folders_of(root, "fit")
}

/// Every stored prediction's folder under `root`.
pub fn predictions(root: &Path) -> Vec<PathBuf> {
    folders_of(root, "prediction")
}

/// The one fit run under `root`; panics naming what is there otherwise.
pub fn the_run(root: &Path) -> PathBuf {
    let mut r = runs(root);
    assert_eq!(r.len(), 1, "expected one fit run under {}: {r:?}", root.display());
    r.pop().unwrap()
}

/// The fit runs under `root` whose method (the `runs/<method>-…` directory)
/// is `method`.
pub fn runs_of_method(root: &Path, method: &str) -> Vec<PathBuf> {
    runs(root)
        .into_iter()
        .filter(|r| {
            let group = r.parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
            group.rsplit_once('-').map(|(m, _)| m) == Some(method)
        })
        .collect()
}

/// The problem folder a run folder sits in.
pub fn problem_of(run: &Path) -> PathBuf {
    run.parent().unwrap().parent().unwrap().parent().unwrap().to_path_buf()
}

/// A run's `run_inputs.json`: its method, backend, seed, chain count, starts,
/// sampler, best log-likelihood and chain, and run and problem ids.
pub fn run_inputs(run: &Path) -> serde_json::Value {
    json(&run.join("run_inputs.json"))
}

/// A run's id, from its `run_inputs.json`.
pub fn run_id(run: &Path) -> String {
    run_inputs(run)["run_id"].as_str().unwrap().to_string()
}

/// The file `path` parsed as JSON.
pub fn json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Copy the directory `from`, which may hold a run store, to `to`,
/// recursively, writing each file's bytes into a new file, so that no
/// modification time is kept on any platform. A reader of the store checks a
/// file whose time differs from its record by its digest.
pub fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::write(&dst, std::fs::read(&src).unwrap()).unwrap();
        }
    }
}
