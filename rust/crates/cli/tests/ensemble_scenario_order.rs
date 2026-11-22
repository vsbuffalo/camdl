//! A combined table follows the order its cells were asked for; the cells do
//! not.
//!
//! `simulate --scenario a,b` and `--scenario b,a` run the same cells: a cell's
//! process seed does not depend on its scenario's position, so each cell has
//! the same stored folder under both orders. Their combined tables differ: the
//! rows follow the order asked for, and the `replicate` column is the row's
//! position. The same holds for `--seeds 1,2` and `--seeds 2,1`. Both orders
//! run into one store must leave one stored cell per distinct cell, and each
//! run's `-o` table must be the one its order asks for, the same on a repeat.
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

mod sim_store_cells;

fn simulate(dir: &Path, cells: &[&str], mirror: &str) -> Output {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    let mut args = vec![
        "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
        "--dt", "1", "--store", "out", "-o", mirror,
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

/// Run `first` then `second` (one cell set in two orders) and `first` again
/// into one store. Each run must exit 0 without an ensemble warning; the two
/// orders write different tables over the same `n_cells` stored cells, and the
/// repeat of `first` writes its table again byte for byte.
fn assert_each_order_is_its_own_table(first: &[&str], second: &[&str], n_cells: usize) {
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

    let one = std::fs::read(dir.join("one.tsv")).unwrap();
    let two = std::fs::read(dir.join("two.tsv")).unwrap();
    assert_ne!(one, two, "the two orders must write different tables for this test to bite");
    assert_eq!(one, std::fs::read(dir.join("one_again.tsv")).unwrap(), "a repeated run is deterministic");
    let cells = sim_store_cells::sim_run_ids(&dir.join("out"));
    assert_eq!(cells.len(), n_cells, "both orders share one stored cell per distinct cell: {cells:?}");
}

#[test]
fn scenario_order_reorders_the_table_over_the_same_cells() {
    assert_each_order_is_its_own_table(
        &["--seed", "3", "--replicates", "2", "--scenario", "a,b"],
        &["--seed", "3", "--replicates", "2", "--scenario", "b,a"],
        4,
    );
}

#[test]
fn seed_order_reorders_the_table_over_the_same_cells() {
    assert_each_order_is_its_own_table(
        &["--seeds", "1,2", "--scenario", "a"],
        &["--seeds", "2,1", "--scenario", "a"],
        2,
    );
}
