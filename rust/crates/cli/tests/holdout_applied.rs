//! gh#585 (Stage 3.1 of the 2026-08-29 honest-predictive-evaluation
//! proposal): a fit that declares `holdout_after` actually trains on the
//! truncated series, and writes the applied training window into
//! `fit.meta.json` — the §3.7.3(b) positive proof `camdl compare`'s
//! non-leakage gate reads. A rerun that hits the cache end-to-end must not
//! erase the recorded window (the sidecar is rewritten before the
//! short-circuit, without re-loading data).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn golden_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/seir_observations.ir.json")
}

fn find_fit_meta(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                return Some(p);
            }
        }
    }
    None
}

#[test]
fn fit_run_applies_holdout_and_records_the_training_window() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("obs.tsv"),
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
    let toml_src = format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data]
holdout_after = 21.0

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
beta     = 0.1

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().canonicalize().unwrap().display());
    std::fs::write(dir.join("fit.toml"), toml_src).unwrap();

    let run = || {
        Command::new(&bin)
            .args(["fit", "run", "fit.toml", "--seed", "1"])
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    };

    let out = run();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "fit run with holdout_after failed:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout));

    // The banner names the training window and what was withheld.
    assert!(stderr.contains("holdout:") && stderr.contains("withheld"),
        "the fit banner must print the training window:\n{stderr}");

    // The §3.7.3(b) proof: fit.meta.json carries the applied window.
    let meta_path = find_fit_meta(&dir.join("results"))
        .expect("fit run wrote a fit.meta.json under results/");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["training_window"]["train_end"], serde_json::json!(21.0),
        "fit.meta.json must record the applied training window: {meta}");

    // An all-cache-hit rerun rewrites the sidecar without re-loading data —
    // the recorded proof must survive (sticky, like `label`).
    let out2 = run();
    assert!(out2.status.success(), "cache-hit rerun failed:\nstderr={}",
        String::from_utf8_lossy(&out2.stderr));
    let meta2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta2["training_window"]["train_end"], serde_json::json!(21.0),
        "a cache-hit rerun must not erase the recorded training window");
}
