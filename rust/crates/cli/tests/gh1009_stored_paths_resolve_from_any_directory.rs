//! gh#1009: a stored fit is read from any directory.
//!
//! A fit problem stores the data it was fitted to, as `inputs/<stream>.tsv`,
//! and its `problem.toml` names those copies. So `camdl compare`, run from
//! the project's parent directory with the runs named by their folder paths,
//! finds the data the fits were fitted to, and still finds it after the file
//! the fit was given has moved.

mod fit_store_support;

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
    // No `CAMDL_STORE`: `fit run` stores into `proj/camdl-store`, and a
    // command run elsewhere finds a run by its folder path alone.
    Command::new(&bin).env_remove("CLICOLOR_FORCE").env_remove("CAMDL_STORE")
        .current_dir(cwd).args(args).output().unwrap()
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
    let runs = fit_store_support::runs(&proj.join("camdl-store"));
    assert_eq!(runs.len(), 2, "{runs:?}");
    // Each problem stores the data it was fitted to, and its problem names
    // the stored copy: the bytes the fit recorded the hash of.
    for run in &runs {
        let problem = fit_store_support::problem_of(run);
        let stored = std::fs::read(problem.join("inputs/cases.tsv")).unwrap();
        assert_eq!(stored, std::fs::read(proj.join("data/cases.tsv")).unwrap());
        let toml = std::fs::read_to_string(problem.join("problem.toml")).unwrap();
        assert!(toml.contains(r#"cases = "inputs/cases.tsv""#), "{toml}");
        let meta = fit_store_support::json(&run.join("fit.meta.json"));
        assert!(meta["data_hashes"]["cases"].is_string(), "{meta}");
    }

    // From the project's parent: the runs named as `proj/camdl-store/…`.
    let rel = |run: &Path| -> String { run.strip_prefix(&outer).unwrap().to_string_lossy().into_owned() };
    let (run_a, run_b) = (rel(&runs[0]), rel(&runs[1]));
    let args = ["compare", run_a.as_str(), run_b.as_str(), "--draws", "1", "--replicates", "1", "--progress", "none"];
    let out = camdl(&outer, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "compare from the parent directory must find the data:\n{stderr}");

    // The file the fits were given moves: the stored copy is what they read.
    std::fs::rename(proj.join("data/cases.tsv"), proj.join("data/moved.tsv")).unwrap();
    let out = camdl(&outer, &args);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "compare reads the data the problem stored:\n{stderr}");
}
