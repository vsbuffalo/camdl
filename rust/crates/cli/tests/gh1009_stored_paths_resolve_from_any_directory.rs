//! gh#1009: a stored fit's config path resolves against the fit, not against
//! the directory a later command runs in.
//!
//! `fit.meta.json` records the producing config as `source_config`, relative
//! to the fit folder, and the archived config's relative data paths resolve
//! against `<fit folder>/<source_config>`. So `camdl compare`, run from the
//! project's parent directory, finds the data the fits were fitted to — where
//! a path recorded relative to `fit run`'s working directory sent it to a file
//! that did not exist. A data file that is genuinely gone is named with where
//! it was looked for and the hash the fit recorded.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn camdl(cwd: &Path, args: &[&str]) -> Output {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    Command::new(&bin).env_remove("CLICOLOR_FORCE").current_dir(cwd).args(args).output().unwrap()
}

/// The single directory under `dir` whose name starts with `prefix`.
fn only(dir: &Path, prefix: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    assert_eq!(hits.len(), 1, "one `{prefix}*` under {}: {hits:?}", dir.display());
    hits.into_iter().next().unwrap()
}

#[test]
fn compare_from_another_directory_finds_the_fits_data() {
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let outer = std::env::temp_dir().join(format!("camdl_gh1009_{}_{ns}", std::process::id()));
    let _guard = Tmp(outer.clone());
    // proj/{model.ir.json, data/cases.tsv, fits/<name>.toml}: the configs one
    // level down, naming the model and data with `../`.
    let proj = outer.join("proj");
    std::fs::create_dir_all(proj.join("fits")).unwrap();
    std::fs::create_dir_all(proj.join("data")).unwrap();
    let fixtures = manifest().join("tests/fixtures/reporting");
    std::fs::copy(fixtures.join("cases.tsv"), proj.join("data/cases.tsv")).unwrap();
    let ir = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(fixtures.join("sir.camdl"))
        .output().unwrap();
    assert!(ir.status.success(), "camdlc: {}", String::from_utf8_lossy(&ir.stderr));
    std::fs::write(proj.join("model.ir.json"), &ir.stdout).unwrap();
    for (name, rho) in [("a", "0.5"), ("b", "0.6")] {
        std::fs::write(proj.join(format!("fits/{name}.toml")), format!(r#"
[model]
camdl = "../model.ir.json"
[data.observations]
cases = "../data/cases.tsv"
[estimate]
beta  = {{ start = 0.5 }}
gamma = {{ start = 0.2 }}
[fixed]
rho = {rho}
N0  = 2000
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 2
particles  = 50
iterations = 300
burn_in    = 100
"#)).unwrap();
        let out = camdl(&proj, &["fit", "run", &format!("fits/{name}.toml"), "--seed", "1", "--progress", "none"]);
        assert!(out.status.success(), "fit {name}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let fits = proj.join("results/fits");
    let seg_a = only(&fits, "a-");
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(seg_a.join("fit.meta.json")).unwrap()).unwrap();
    assert_eq!(meta["schema_version"], 2, "{meta}");
    assert_eq!(meta["source_config"], "../../../fits/a.toml", "relative to the fit folder: {meta}");
    assert_eq!(meta["source_model"], "../../../model.ir.json", "relative to the fit folder: {meta}");
    assert!(meta.get("fit_toml_path").is_none() && meta.get("model_path").is_none(), "{meta}");

    // From the project's parent: the leaves named as `proj/results/…`.
    let rel = |seg: &Path| -> String {
        let leaf = only(&only(seg, "pmmh-"), "seed_");
        leaf.strip_prefix(&outer).unwrap().to_string_lossy().into_owned()
    };
    let (leaf_a, leaf_b) = (rel(&seg_a), rel(&only(&fits, "b-")));
    let args = ["compare", leaf_a.as_str(), leaf_b.as_str(), "--draws", "1", "--replicates", "1", "--progress", "none"];
    let out = camdl(&outer, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "compare from the parent directory must find the data:\n{stderr}");

    // The data moved: the error names the stream, where it looked, and the
    // hash the fit recorded for the file.
    std::fs::rename(proj.join("data/cases.tsv"), proj.join("data/moved.tsv")).unwrap();
    let out = camdl(&outer, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the data is gone:\n{stderr}");
    let recorded = meta["data_hashes"]["cases"].as_str().unwrap();
    assert!(stderr.contains("stream `cases`: looked for proj/results/fits/")
        && stderr.contains("/../../../fits/../data/cases.tsv")
        && stderr.contains(&format!("recorded hash {recorded}")),
        "names the stream, the path looked for and the recorded hash:\n{stderr}");
}
