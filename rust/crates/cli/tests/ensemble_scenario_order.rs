//! One ensemble id stands for one combined table.
//!
//! `simulate --scenario a,b` and `--scenario b,a` run the same cells: a cell's
//! process seed does not depend on its scenario's position, so each cell has
//! the same `Sim` leaf under both orders. Their combined tables differ: the
//! rows follow the order asked for, and the `replicate` column is the row's
//! position. The same holds for `--seeds 1,2` and `--seeds 2,1`. Each run must
//! store, under its own ensemble id, exactly the table it produced; a run that
//! finds a different table under its id, warns and exits 0 leaves one id
//! naming two tables.
//!
//! Shells out to the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate in [0.01, 5.0]
  gamma : rate in [0.01, 5.0]
}

init { S = 990  I = 10  R = 0 }

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 20 'days }

scenarios {
  a { set = { beta = 0.3 } }
  b { set = { beta = 0.6 } }
}
"#;

const PARAMS: &str = "beta = 0.4\ngamma = 0.2\n";

fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    out
}

fn simulate(dir: &Path, cells: &[&str], mirror: &str) -> Output {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    let mut args = vec![
        "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
        "--dt", "1", "--output-dir", "out", "-o", mirror,
    ];
    args.extend_from_slice(cells);
    Command::new(bin)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(dir)
        .args(&args)
        .output()
        .expect("spawn camdl")
}

/// Every ensemble leaf in the store, by `run_id`, with its stored table.
fn ensembles(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    run_leaves(&dir.join("out").join("ensembles"))
        .into_iter()
        .map(|leaf| {
            let rec: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
            let table = std::fs::read(leaf.join("ensemble.tsv")).unwrap();
            (rec["run_id"].as_str().unwrap().to_string(), table)
        })
        .collect()
}

/// Run `first` then `second` (one cell set in two orders) and `first` again
/// into one store. Each run must exit 0 without an ensemble warning and leave
/// its own table stored under an id of its own; the repeat of `first` must
/// find its table already stored.
fn assert_each_order_is_its_own_ensemble(first: &[&str], second: &[&str]) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("p.toml"), PARAMS).unwrap();

    for (cells, mirror) in [(first, "one.tsv"), (second, "two.tsv"), (first, "one_again.tsv")] {
        let out = simulate(dir, cells, mirror);
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{cells:?} failed:\n{err}");
        assert!(!err.contains("warning: ensemble"), "{cells:?} warned about its ensemble:\n{err}");
    }

    let stored = ensembles(dir);
    let one = std::fs::read(dir.join("one.tsv")).unwrap();
    let two = std::fs::read(dir.join("two.tsv")).unwrap();
    assert_ne!(one, two, "the two orders must write different tables for this test to bite");
    assert_eq!(one, std::fs::read(dir.join("one_again.tsv")).unwrap(), "a repeated run is deterministic");
    assert_eq!(stored.len(), 2, "two tables, two ensemble ids: {:?}", stored.keys());
    for (mirror, table) in [("one.tsv", &one), ("two.tsv", &two)] {
        assert!(
            stored.values().any(|t| t == table),
            "the table {mirror} produced is not stored under any ensemble id"
        );
    }
}

#[test]
fn scenario_order_never_gives_one_ensemble_id_two_tables() {
    assert_each_order_is_its_own_ensemble(
        &["--seed", "3", "--replicates", "2", "--scenario", "a,b"],
        &["--seed", "3", "--replicates", "2", "--scenario", "b,a"],
    );
}

#[test]
fn seed_order_never_gives_one_ensemble_id_two_tables() {
    assert_each_order_is_its_own_ensemble(
        &["--seeds", "1,2", "--scenario", "a"],
        &["--seeds", "2,1", "--scenario", "a"],
    );
}
