//! `camdl profile` against an INDEXED, LONG-FORM observation family — the
//! shape a real stratified survey fit uses (e.g. malaria prevalence indexed by
//! village × age, one long TSV `(time, village, age, n_positive, n_examined)`).
//!
//! The `sir_two_patch_long_obs` fixture declares a stratified header
//! `cases[p in patch]` (patches urban, rural) → two IR leaves `cases_urban`,
//! `cases_rural`, each carrying a `stratum = [("patch", <level>)]` selector, all
//! sharing `source = "cases"`. A SINGLE long-form file `(time, patch, cases)`
//! binds to the family root; the long-form loader routes each row to the leaf
//! whose stratum matches the row's `patch` value BY NAME.
//!
//! `camdl fit run` handles this (source-based binding + per-stratum slicing via
//! `load_observations`). Before this fix `camdl profile` did NOT: it bound by
//! exact stream NAME (so the fit-toml `[data.observations] cases = FILE` form
//! failed to resolve the family root `cases` to its leaves) and it loaded via a
//! wide-column reader (so the CLI `--data cases=FILE` form fed BOTH patches'
//! rows into EACH stream — a silent-wrong per-stream scoring). These tests pin
//! both binding paths end-to-end through the real binary.

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
/// (the pre-fix wide-loader bug) it sees duplicate times → the strictly
/// increasing check rejects it; a correctly SLICED stream sees each time
/// exactly once.
fn write_long_form(path: &Path) {
    write(path,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\t6\n\
         21\turban\t25\n21\trural\t9\n");
}

/// Collect the per-grid-point best loglik from the `profile_point` run.json
/// leaves under `<out_root>/profiles/` (same layout profile_multi_stream reads).
fn collect_logliks(out_root: &Path) -> Vec<f64> {
    fn walk(dir: &Path, out: &mut Vec<(String, f64)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("profile_point") {
                    let point = v.get("levels").and_then(|ls| ls.as_array())
                        .and_then(|a| a.iter().find(|l|
                            l.get("name").and_then(|n| n.as_str()) == Some("point")))
                        .and_then(|l| l.get("label").and_then(|x| x.as_str()))
                        .unwrap_or("").to_string();
                    if let Some(ll) = v.get("inputs")
                        .and_then(|i| i.get("best_loglik")).and_then(|x| x.as_f64()) {
                        out.push((point, ll));
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), out); } }
        }
    }
    let mut pairs = Vec::new();
    walk(&out_root.join("profiles"), &mut pairs);
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs.into_iter().map(|(_, ll)| ll).collect()
}

/// CLI binding path: `--data cases=FILE` on the long-form file. Pre-fix the
/// wide-column loader fed BOTH patches' rows into each stream → "non-increasing
/// observation times (7 then 7)". Post-fix each leaf is sliced to its own patch.
#[test]
fn profile_cli_family_binding_slices_per_stratum() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_long_form(&data);
    let out_root = tmp.path().join("out_cli");
    let out_tsv = tmp.path().join("profile_cli.tsv");

    let output = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            "--scenario", "baseline",
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--sweep", "beta=lin(0.04,0.06,2)",
            "--fixed", "gamma=0.1", "--fixed", "rho=0.6", "--fixed", "k=5.0",
            "--particles", "100", "--iterations", "1", "--starts", "1",
            "--rw-sd", "auto", "--seed", "1",
            "--output", &out_tsv.to_string_lossy(),
        ])
        .output()
        .expect("spawn camdl profile");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(),
        "profile (CLI family binding) must succeed on long-form data:\nstderr=\n{stderr}");
    assert!(stderr.contains("cases_urban") && stderr.contains("cases_rural"),
        "both stratum leaves must bind from the long-form file:\n{stderr}");
    assert!(!stderr.contains("non-increasing"),
        "each leaf must be SLICED to its own patch (no cross-patch time collision):\n{stderr}");

    let ll = collect_logliks(&out_root);
    assert_eq!(ll.len(), 2, "expected 2 grid points, got {ll:?}");
    for (i, v) in ll.iter().enumerate() {
        assert!(v.is_finite() && *v < 0.0,
            "grid {i} loglik must be finite negative, got {v}");
    }
}

/// Fit-toml binding path (garki's exact path): `[data.observations] cases =
/// FILE`, NO `--data`. Pre-fix `effective_observations` returned the raw key
/// `cases` and profile exact-matched it against the IR stream names
/// (`cases_urban`/`cases_rural`) → "bound stream 'cases' has no matching IR
/// observation block". Post-fix the family root resolves by `source` to both
/// leaves.
#[test]
fn profile_fit_toml_family_root_resolves_to_strata() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_long_form(&data);

    let toml = tmp.path().join("prof.toml");
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

    let out_root = tmp.path().join("out_fit");
    let out_tsv = tmp.path().join("profile_fit.tsv");
    let output = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            "--scenario", "baseline",
            "--fit", &toml.to_string_lossy(),
            "--sweep", "beta=lin(0.04,0.06,2)",
            "--particles", "100", "--iterations", "1", "--starts", "1",
            "--rw-sd", "auto", "--seed", "1",
            "--output", &out_tsv.to_string_lossy(),
        ])
        .output()
        .expect("spawn camdl profile");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(),
        "profile (fit-toml family root) must resolve `cases` to its strata:\nstderr=\n{stderr}");
    assert!(!stderr.contains("no matching IR"),
        "family root `cases` must fan out by source, not exact-match a stream name:\n{stderr}");
    assert!(stderr.contains("cases_urban") && stderr.contains("cases_rural"),
        "both stratum leaves must bind from the fit-toml [data.observations]:\n{stderr}");

    let ll = collect_logliks(&out_root);
    assert_eq!(ll.len(), 2, "expected 2 grid points, got {ll:?}");
    for (i, v) in ll.iter().enumerate() {
        assert!(v.is_finite() && *v < 0.0,
            "grid {i} loglik must be finite negative, got {v}");
    }
}
