//! §4.2 by-name level matching — end-to-end smoke for the LONG-FORM
//! stratified-observation data path (2026-06-10 observation data-entry).
//!
//! The `sir_two_patch_long_obs` fixture declares a stratified observation
//! header `cases[p in patch]` (patches urban, rural). It expands to two IR
//! leaves (`cases_urban`, `cases_rural`), each carrying a structured
//! `stratum = [{dim:"patch", level:<lv>}]` selector. A SINGLE long-form data
//! file `(time, patch, cases)` is bound via the family-root form
//! `--data cases=FILE`, and the long-form loader routes each row to the leaf
//! whose stratum matches the row's `patch` value BY NAME.
//!
//! This pins three properties end-to-end through the real binary:
//!   1. the family binds + scores a finite loglik (rows routed by name);
//!   2. an `NA` value is a HOLE — skipped, not scored;
//!   3. the hole's loglik differs from scoring an observed 0 at the same
//!      cell (so a hole is genuinely absent, never a false zero).

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
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn long_obs_model() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/sir_two_patch_long_obs.ir.json")
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents).expect("write fixture");
}

fn run_pfilter(bin: &Path, model: &Path, data: &Path, params: &Path) -> (bool, String, String) {
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &model.to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--particles", "300", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn long_form_family_binds_routes_and_skips_hole() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    let params = tmp.path().join("params.toml");
    write(&params, "beta = 0.05\ngamma = 0.1\nrho = 0.6\nk = 5.0\n");

    // Long-form file: interleaved patch rows. Rural at t=14 is a HOLE (`NA`).
    let holed = tmp.path().join("cases_holed.tsv");
    write(&holed,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\tNA\n\
         21\turban\t25\n21\trural\t6\n");

    let (ok, stdout, stderr) = run_pfilter(&bin, &long_obs_model(), &holed, &params);
    assert!(ok, "long-form pfilter failed:\nstdout={stdout}\nstderr={stderr}");

    // Both leaves of the family bound from the ONE file (routed by name).
    assert!(stderr.contains("cases_urban") && stderr.contains("cases_rural"),
        "both stratum leaves must bind from the long-form file: {stderr}");

    let ll_hole: f64 = stdout.trim().parse()
        .unwrap_or_else(|_| panic!("loglik parse from stdout: {stdout:?}"));
    assert!(ll_hole.is_finite() && ll_hole < 0.0,
        "expected finite negative loglik, got {ll_hole}");

    // Same file, but the rural t=14 cell is an observed 0 instead of a hole.
    let zeroed = tmp.path().join("cases_zero.tsv");
    write(&zeroed,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\t0\n\
         21\turban\t25\n21\trural\t6\n");
    let (ok2, stdout2, stderr2) = run_pfilter(&bin, &long_obs_model(), &zeroed, &params);
    assert!(ok2, "observed-0 pfilter failed:\nstdout={stdout2}\nstderr={stderr2}");
    let ll_zero: f64 = stdout2.trim().parse()
        .unwrap_or_else(|_| panic!("loglik parse: {stdout2:?}"));

    // A hole is NOT an observed zero — the hole skips its term, so the two
    // logliks must differ. (Same seed/particles/params; the only change is
    // hole vs observed-0 at the rural t=14 cell.)
    assert!((ll_hole - ll_zero).abs() > 1e-6,
        "a hole must be SKIPPED, not scored as 0: hole loglik {ll_hole} == \
         observed-0 loglik {ll_zero}");
}

#[test]
fn long_form_unknown_level_is_a_located_error() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let params = tmp.path().join("params.toml");
    write(&params, "beta = 0.05\ngamma = 0.1\nrho = 0.6\nk = 5.0\n");

    // `suburban` is not a model patch level → E281, located.
    let bad = tmp.path().join("cases_bad.tsv");
    write(&bad,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\tsuburban\t3\n");
    let (ok, stdout, stderr) = run_pfilter(&bin, &long_obs_model(), &bad, &params);
    assert!(!ok, "an unknown level must fail:\nstdout={stdout}\nstderr={stderr}");
    assert!(stderr.contains("E281"), "must be E281: {stderr}");
    assert!(stderr.contains("suburban"), "must name the offending level: {stderr}");
    assert!(stderr.contains("urban") && stderr.contains("rural"),
        "must list the valid levels [urban, rural]: {stderr}");
}
