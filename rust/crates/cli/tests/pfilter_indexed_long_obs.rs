//! `camdl pfilter` against an INDEXED, LONG-FORM observation family — the same
//! stratified-survey shape `camdl fit run` and `camdl profile` handle. This pins
//! that pfilter binds the family through the shared by-source seam, not by exact
//! leaf name.
//!
//! The `sir_two_patch_long_obs` fixture declares a stratified header
//! `cases[p in patch]` (patches urban, rural) → two IR leaves `cases_urban`,
//! `cases_rural`, each carrying a `stratum = [("patch", <level>)]` selector, all
//! sharing `source = "cases"`. A SINGLE long-form file `(time, patch, cases)`
//! binds to the family root; the long-form loader routes each row to the leaf
//! whose stratum matches the row's `patch` value BY NAME.
//!
//! Two binding paths:
//!   - CLI `--data cases=FILE` — `resolve_data_specs` expands `cases` to leaf
//!     names, which already bound before the seam consolidation.
//!   - fit-toml `[data.observations] cases = FILE` — `effective_observations`
//!     returns the raw key `cases`, which pfilter exact-matched against the IR
//!     stream names (`cases_urban`/`cases_rural`) → "no observation block named
//!     'cases'". The seam resolves the family root by `source` to both leaves.

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

/// A long-form `(time, patch, cases)` file with urban and rural rows
/// interleaved at the SAME times. If a stream is fed both patches' rows
/// (the wide-loader bug) it sees duplicate times → the strictly-increasing
/// check rejects it; a correctly SLICED stream sees each time exactly once.
fn write_long_form(path: &Path) {
    write(path,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\t6\n\
         21\turban\t25\n21\trural\t9\n");
}

fn params_file(tmp: &Path) -> PathBuf {
    let p = tmp.join("params.toml");
    write(&p, "beta = 0.05\ngamma = 0.1\nrho = 0.6\nk = 5.0\n");
    p
}

/// CLI binding path: `--data cases=FILE` on the long-form file. The family root
/// `cases` expands to leaf names; each leaf is sliced to its own patch. This
/// already worked before the seam consolidation — a regression guard.
#[test]
fn pfilter_cli_family_binding_slices_per_stratum() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_long_form(&data);
    let params = params_file(tmp.path());

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("out_cli"))
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--particles", "100", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "pfilter (CLI family binding) must succeed on long-form data:\nstdout={stdout}\nstderr={stderr}");
    assert!(stderr.contains("cases_urban") && stderr.contains("cases_rural"),
        "both stratum leaves must bind from the long-form file:\n{stderr}");
    assert!(!stderr.contains("non-increasing"),
        "each leaf must be SLICED to its own patch (no cross-patch time collision):\n{stderr}");
    let ll: f64 = stdout.trim().parse()
        .unwrap_or_else(|_| panic!("loglik parse from stdout: {stdout:?}\nstderr={stderr}"));
    assert!(ll.is_finite() && ll < 0.0,
        "expected finite negative loglik, got {ll}");
}

/// Fit-toml binding path (garki's exact path): `[data.observations] cases =
/// FILE`, NO `--data`. Before the seam consolidation `effective_observations`
/// returned the raw key `cases` and pfilter exact-matched it against the IR
/// stream names (`cases_urban`/`cases_rural`) → "no observation block named
/// 'cases'". The seam resolves the family root by `source` to both leaves.
#[test]
fn pfilter_fit_toml_family_root_resolves_to_strata() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_long_form(&data);
    let params = params_file(tmp.path());

    let toml = tmp.path().join("pf.toml");
    write(&toml, &format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ log_normal = {{ mu = -2.3, sigma = 0.5 }} }} }}
[fixed]
rho = 0.6
k = 5.0
[stages.dummy]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 1
cooling    = 0.5
"#,
        out  = tmp.path().join("results").display(),
        ir   = long_obs_model().display(),
        data = data.display(),
    ));

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("out_fit"))
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--fit", &toml.to_string_lossy(),
            "--particles", "100", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "pfilter (fit-toml family root) must resolve `cases` to its strata:\nstdout={stdout}\nstderr={stderr}");
    assert!(!stderr.contains("no observation block named") && !stderr.contains("no matching IR"),
        "family root `cases` must fan out by source, not exact-match a stream name:\n{stderr}");
    assert!(stderr.contains("cases_urban") && stderr.contains("cases_rural"),
        "both stratum leaves must bind from the fit-toml [data.observations]:\n{stderr}");
    let ll: f64 = stdout.trim().parse()
        .unwrap_or_else(|_| panic!("loglik parse from stdout: {stdout:?}\nstderr={stderr}"));
    assert!(ll.is_finite() && ll < 0.0,
        "expected finite negative loglik, got {ll}");
}
