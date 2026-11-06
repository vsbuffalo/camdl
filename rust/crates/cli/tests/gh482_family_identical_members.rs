//! gh#482: an indexed intervention family whose members are identical applied
//! its action once per member. `sia[r in round] : transfer(count = n, …)`, with
//! a body that never mentions `r`, compiled with no diagnostic and moved 2n
//! individuals (S = 900, V = 100 at t = 11 with n = 50). The compiler now
//! refuses it with E415; this test pins the text a user sees through the CLI,
//! and a positive control pins that the unindexed form still moves n.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
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

fn model(head: &str) -> String {
    format!(
        r#"
dimensions {{ round = [r1, r2] }}
compartments {{ S, V }}
parameters {{ n : count }}
transitions {{ }}
init {{ S = 1000 V = 0 }}
interventions {{
  {head} : transfer(count = n, from = S, to = V) at [10]
}}
simulate {{ from = 0 'days to = 20 'days }}
"#
    )
}

fn simulate(bin: &Path, model_path: &Path) -> std::process::Output {
    Command::new(bin)
        .args([
            "simulate",
            &model_path.to_string_lossy(),
            "--param",
            "n=50",
            "--enable",
            "sia",
            "--stdout",
            "--backend",
            "chain_binomial",
            "--dt",
            "1",
        ])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl simulate")
}

#[test]
fn family_ignoring_its_binder_is_refused_with_e415() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("m.camdl");
    std::fs::write(&path, model("sia[r in round]")).unwrap();

    let out = simulate(&bin, &path);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a family of identical members must be refused, not simulated. stderr:\n{stderr}"
    );
    for needle in [
        "E415",
        "intervention family 'sia[r in round]' expands to 2 instances (sia_r1, sia_r2)",
        "its action is applied 2 times",
        "binder 'r' does not distinguish the instances",
    ] {
        assert!(
            stderr.contains(needle),
            "expected {needle:?} in the diagnostic, got:\n{stderr}"
        );
    }
}

#[test]
fn unindexed_intervention_moves_n_once() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("m.camdl");
    std::fs::write(&path, model("sia")).unwrap();

    let out = simulate(&bin, &path);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "simulate failed:\n{stderr}");

    // A `# <version>` provenance line precedes the TSV header.
    let mut lines = stdout.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no column {name} in {header:?}"))
    };
    let (t, s, v) = (col("t"), col("S"), col("V"));
    let row: Vec<&str> = lines
        .map(|l| l.split('\t').collect::<Vec<_>>())
        .find(|r| r[t].parse::<f64>().unwrap() == 11.0)
        .expect("a row at t = 11");
    assert_eq!(row[s].parse::<f64>().unwrap(), 950.0, "S at t = 11");
    assert_eq!(row[v].parse::<f64>().unwrap(), 50.0, "V at t = 11");
}
