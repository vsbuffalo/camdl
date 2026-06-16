//! gh#156 e2e: the trajectory output view — `--output-every` / `--no-flows` /
//! `--columns`. Exercises the real `simulate` write path (the `--stdout`
//! mirror, which drains the same filtered `StreamSink` buffer the CAS leaf and
//! `-o` mirror use).
//!
//! Silent-skip if the release `camdl` binary is not built (same convention as
//! the other e2e tests; gh#105) so a plain `cargo test` without a release build
//! does not fail.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> Option<PathBuf> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let p = Path::new(manifest).join("../../target/release/camdl");
    if p.exists() { Some(p) } else { None }
}

/// A self-contained golden: S/I/R/W compartments + three flows
/// (infection/recovery/waning), with the model's own weekly output cadence.
fn model() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("../../../ir/golden/cholera_siwr.ir.json")
}

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// The trajectory header columns from a `--stdout` run (skips the leading
/// `# version` comment line).
fn header(stdout: &[u8]) -> Vec<String> {
    let s = String::from_utf8_lossy(stdout);
    let line = s.lines().find(|l| !l.starts_with('#')).unwrap_or("");
    line.split('\t').map(|c| c.to_string()).collect()
}

/// Count data rows (skip the `# version` comment and the `t\t…` header).
fn data_rows(stdout: &[u8]) -> usize {
    let s = String::from_utf8_lossy(stdout);
    s.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("t\t"))
        .count()
}

#[test]
fn no_flows_drops_only_flow_columns() {
    let Some(camdl) = camdl_bin() else { return };
    let m = model();
    let m = m.to_str().unwrap();

    let def = run(&camdl, &["simulate", m, "--stdout"]);
    assert!(def.status.success(), "default sim failed: {}",
        String::from_utf8_lossy(&def.stderr));
    let dh = header(&def.stdout);
    assert!(dh.iter().any(|c| c.starts_with("flow_")),
        "default header should carry flow_ columns: {:?}", dh);

    let nf = run(&camdl, &["simulate", m, "--stdout", "--no-flows"]);
    assert!(nf.status.success());
    let nh = header(&nf.stdout);
    assert!(!nh.iter().any(|c| c.starts_with("flow_")),
        "--no-flows must drop every flow_ column: {:?}", nh);
    // The compartment columns are untouched.
    let expect: Vec<&String> = dh.iter().filter(|c| !c.starts_with("flow_")).collect();
    let got: Vec<&String> = nh.iter().collect();
    assert_eq!(expect, got, "--no-flows must keep the compartment columns intact");
}

#[test]
fn columns_restricts_to_the_allow_list() {
    let Some(camdl) = camdl_bin() else { return };
    let m = model();
    let m = m.to_str().unwrap();

    let out = run(&camdl, &["simulate", m, "--stdout", "--columns", "S,flow_infection"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let h = header(&out.stdout);
    let h: Vec<&str> = h.iter().map(String::as_str).collect();
    assert_eq!(h, ["t", "S", "flow_infection"],
        "--columns must restrict to `t` + the listed columns (mixed compartment + flow)");
}

#[test]
fn unknown_column_is_a_hard_error() {
    let Some(camdl) = camdl_bin() else { return };
    let m = model();
    let m = m.to_str().unwrap();

    let out = run(&camdl, &["simulate", m, "--stdout", "--columns", "not_a_real_col"]);
    assert!(!out.status.success(), "an unknown --columns name must fail");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown column") && err.contains("Valid columns are"),
        "the error must name the bad column and list the valid ones: {}", err);
}

#[test]
fn output_every_overrides_the_model_cadence() {
    let Some(camdl) = camdl_bin() else { return };
    let m = model();
    let m = m.to_str().unwrap();

    let def = run(&camdl, &["simulate", m, "--stdout"]);
    let dn = data_rows(&def.stdout);
    // The model declares weekly output; a coarser override must reduce the rows.
    let coarse = run(&camdl, &["simulate", m, "--stdout", "--output-every", "60"]);
    assert!(coarse.status.success(), "{}", String::from_utf8_lossy(&coarse.stderr));
    let cn = data_rows(&coarse.stdout);
    assert!(cn < dn, "a coarser --output-every must reduce rows: {} !< {}", cn, dn);
}
