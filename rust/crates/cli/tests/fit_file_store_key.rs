//! A command given `--fit fit.toml` stores its run in the store the fit
//! file's `store` key names, as `fit run fit.toml` does: `--store`, then the
//! fit file's `store` key (anchored at the file), then `CAMDL_STORE`, then
//! `./camdl-store`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate        in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate        in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  rho   : probability in [0.1, 0.9]
}

transitions {
  infection : S --> I  @ beta * S * I / (S + I + R)
  recovery  : I --> R  @ gamma * I
}

init {
  S = 990
  I = 10
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ poisson(rate = rho * projected)
  }
}

simulate {
  from = 0 'days
  to   = 28 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t12\n14\t30\n21\t41\n28\t25\n";

const FIT: &str = r#"store = "fit-store"

[model]
camdl = "../sir.camdl"

[data.observations]
weekly_cases = "../weekly_cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], start = 0.4 }

[fixed]
gamma = 0.2
rho = 0.6

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 2
cooling = 0.5
"#;

/// A directory holding the model and data, and the fit file one level down in
/// `fits/`, so the key `store = "fit-store"` resolves to `fits/fit-store`.
fn setup() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let d = dir.path();
    std::fs::write(d.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(d.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::create_dir(d.join("fits")).unwrap();
    std::fs::write(d.join("fits/fit.toml"), FIT).unwrap();
    std::fs::write(d.join("theta.toml"), "beta = 0.4\ngamma = 0.2\nrho = 0.6\n").unwrap();
    dir
}

fn camdl(dir: &Path, args: &[&str]) {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let out = Command::new(bin)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("env-store"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("spawn camdl");
    assert!(
        out.status.success(),
        "camdl {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn is_store(root: &Path) -> bool {
    root.join("store.json").is_file()
}

#[test]
fn pfilter_stores_in_the_fit_files_store() {
    let dir = setup();
    let d = dir.path();
    camdl(d, &["pfilter", "sir.camdl", "--fit", "fits/fit.toml", "--params", "theta.toml",
               "--particles", "50", "--seed", "1"]);
    assert!(is_store(&d.join("fits/fit-store")), "pfilter must store in the fit file's `store`");
    assert!(!is_store(&d.join("env-store")), "the fit file's `store` comes above CAMDL_STORE");
}

#[test]
fn pfilter_store_flag_comes_above_the_fit_files_store() {
    let dir = setup();
    let d = dir.path();
    camdl(d, &["pfilter", "sir.camdl", "--fit", "fits/fit.toml", "--params", "theta.toml",
               "--particles", "50", "--seed", "1", "--store", "flag-store"]);
    assert!(is_store(&d.join("flag-store")), "--store comes first");
    assert!(!is_store(&d.join("fits/fit-store")));
}

#[test]
fn pfilter_without_a_store_key_stores_in_camdl_store() {
    let dir = setup();
    let d = dir.path();
    std::fs::write(d.join("fits/fit.toml"), FIT.replace("store = \"fit-store\"\n", "")).unwrap();
    camdl(d, &["pfilter", "sir.camdl", "--fit", "fits/fit.toml", "--params", "theta.toml",
               "--particles", "50", "--seed", "1"]);
    assert!(is_store(&d.join("env-store")), "CAMDL_STORE applies when the fit file names no store");
    assert!(!is_store(&d.join("camdl-store")), "CAMDL_STORE comes above ./camdl-store");
}

#[test]
fn fit_run_store_flag_comes_above_the_fit_files_store() {
    let dir = setup();
    let d = dir.path();
    camdl(d, &["fit", "run", "fits/fit.toml", "--store", "flag-store", "--progress", "none"]);
    assert!(is_store(&d.join("flag-store")), "--store comes first");
    assert!(!is_store(&d.join("fits/fit-store")), "--store comes above the fit file's `store`");
}

/// `fit run` and `pfilter --fit` of one fit file store in one store.
#[test]
fn fit_run_and_pfilter_of_one_fit_file_share_its_store() {
    let dir = setup();
    let d = dir.path();
    camdl(d, &["fit", "run", "fits/fit.toml", "--progress", "none"]);
    camdl(d, &["pfilter", "sir.camdl", "--fit", "fits/fit.toml", "--params", "theta.toml",
               "--particles", "50", "--seed", "1"]);
    let root = d.join("fits/fit-store");
    assert!(is_store(&root), "fit run stores in the fit file's `store`");
    let kinds = stored_kinds(&root);
    assert!(kinds.contains(&"fit".to_string()) && kinds.contains(&"pfilter".to_string()),
        "the fit problem and the filter run share the store; kinds stored: {kinds:?}");
    assert!(!is_store(&d.join("env-store")));
}

/// The kind of every result under `root`, from each result's `identity.json`.
fn stored_kinds(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for path in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            if let Ok(text) = std::fs::read_to_string(path.join("identity.json")) {
                let v: serde_json::Value = serde_json::from_str(&text).unwrap();
                if let Some(kind) = v["kind"].as_str() {
                    out.push(kind.to_string());
                }
            }
            stack.push(path);
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn simulate_stores_in_the_fit_files_store() {
    let dir = setup();
    let d = dir.path();
    camdl(d, &["simulate", "sir.camdl", "--draws", "prior", "--fit", "fits/fit.toml",
               "--n-draws", "2", "--seed", "1"]);
    assert!(is_store(&d.join("fits/fit-store")), "simulate must store in the fit file's `store`");
    assert!(!is_store(&d.join("env-store")), "the fit file's `store` comes above CAMDL_STORE");
}
