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

/// The best loglik of each profile point stored under `store`, in order of
/// the focal `beta` value (read from each point's `identity.json` and
/// `provenance.json`).
fn collect_logliks(store: &Path) -> Vec<f64> {
    fn walk(dir: &Path, out: &mut Vec<(f64, f64)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("identity.json")) {
            let v: serde_json::Value = serde_json::from_str(&b).unwrap();
            if v["kind"] == "profile_point" {
                let focal = v["identity"]["focal"]["beta"].as_f64().expect("focal beta");
                let prov: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("provenance.json")).unwrap()).unwrap();
                out.push((focal, prov["best_loglik"].as_f64().expect("best_loglik")));
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() {
                if e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                    walk(&e.path(), out);
                }
            }
        }
    }
    let mut pairs = Vec::new();
    walk(store, &mut pairs);
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs.into_iter().map(|(_, ll)| ll).collect()
}

/// The long-form file binds to the model's streams through the fit file's
/// `[data.observations]` alone: `--data cases=FILE` on the command line is
/// refused, naming the issue that tracks it, and no point is stored.
#[test]
fn profile_data_flag_on_a_family_binding_is_refused() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_long_form(&data);
    let toml = tmp.path().join("prof.toml");
    write(&toml, &format!(r#"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.1 }}
[fixed]
rho = 0.6
k = 5.0
"#,
        ir   = long_obs_model().display(),
        data = data.display(),
    ));
    let out_root = tmp.path().join("out_cli");

    let output = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            "--fit", &toml.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--sweep", "beta=lin(0.04,0.06,2)",
            "--particles", "100", "--iterations", "1", "--n-starts", "1",
            "--rw-sd", "auto", "--seed", "1",
        ])
        .args(["--store", &out_root.to_string_lossy()])
        .output()
        .expect("spawn camdl profile");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "--data beside --fit must be refused:\n{stderr}");
    assert!(stderr.contains("gh#1354"), "the refusal must name gh#1354:\n{stderr}");
    assert!(collect_logliks(&out_root).is_empty(), "a refused profile must store no point");
}

/// Fit-toml binding path (garki's exact path): `[data.observations] cases =
/// FILE`. Pre-fix `effective_observations` returned the raw key
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
store = "{out}"
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
[method]
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
    let output = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            // No `--scenario baseline`: it sets `gamma`, which this fit
            // estimates, and profile refuses that as `fit run` does. The
            // fit config supplies every value (gamma's start, [fixed]).
            "--fit", &toml.to_string_lossy(),
            "--sweep", "beta=lin(0.04,0.06,2)",
            "--particles", "100", "--iterations", "1", "--n-starts", "1",
            "--rw-sd", "auto", "--seed", "1",
            "--output", &out_tsv.to_string_lossy(),
        ])
        .args(["--store", &out_root.to_string_lossy()])
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
