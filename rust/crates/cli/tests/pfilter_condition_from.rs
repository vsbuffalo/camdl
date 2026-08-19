//! `camdl pfilter` / `camdl profile` honour the conditioning window (gh#621).
//!
//! `condition_from` was a fit-path-only key: `FitRunConfig::build` resolved it,
//! prepended the leading reset-only hole, and ran the W329 wide-first-window
//! enforcer — while `pfilter` and `profile`, the fixed-θ scorers whose whole
//! purpose is a loglik comparable to the fit's, silently dropped it (even when
//! reading the very fit.toml that declares it) and scored the first incidence
//! bin over the whole span from the origin. A −inf from such a run is
//! ambiguous: a bad θ, or just the unconstrainable leading window.
//!
//! Pinned here:
//! - an unconditioned wide first window on an incidence stream is the same
//!   hard error `fit run` raises (W329), naming the fix;
//! - `--condition-from` applies the warm-up (stderr names the window, the
//!   loglik is finite);
//! - the `--fit` toml's `condition_from` is honoured and produces the
//!   byte-identical loglik to the flag form at the same seed.

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

/// Long-form `(time, patch, cases)` whose FIRST observation sits at t = 42 on a
/// 7-day cadence: a 42-day first window against a 7-day cadence (ratio 6 > the
/// W329 threshold 5) — the gh#134 wrong-number shape on an incidence stream.
fn write_wide_window_data(path: &Path) {
    write(path,
        "time\tpatch\tcases\n\
         42\turban\t12\n42\trural\t3\n\
         49\turban\t18\n49\trural\t6\n\
         56\turban\t25\n56\trural\t9\n\
         63\turban\t20\n63\trural\t7\n");
}

fn params_file(tmp: &Path) -> PathBuf {
    let p = tmp.join("params.toml");
    // Supercritical (R0 = 3): the epidemic must still be producing cases at
    // the late first observation (t = 42), or every particle scores the
    // 12-case bin against a dead epidemic and the loglik is a legitimate -inf.
    write(&p, "beta = 0.3\ngamma = 0.1\nrho = 0.6\nk = 5.0\n");
    p
}

fn run_loglik(out: &std::process::Output) -> f64 {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    stdout.trim().parse().unwrap_or_else(|_| {
        panic!("loglik parse from stdout: {stdout:?}\nstderr={stderr}")
    })
}

/// A minimal schema-complete fit toml. `condition_from` is optional so the
/// unconditioned-W329 tests can share it.
fn write_fit_toml(tmp: &Path, data: &Path, condition_from: Option<&str>) -> PathBuf {
    let toml = tmp.join("pf.toml");
    let cond = condition_from
        .map(|c| format!("condition_from = \"{c}\"\n"))
        .unwrap_or_default();
    write(&toml, &format!(r#"
output_dir = "{out}"
{cond}[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ log_normal = {{ mu = -2.3, sigma = 0.5 }} }} }}
[fixed]
beta = 0.3
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
        out  = tmp.join("results").display(),
        ir   = long_obs_model().display(),
        data = data.display(),
    ));
    toml
}

/// No conditioning, wide first window, incidence stream: the first datum
/// cannot constrain the whole 42-day span. `fit run` hard-errors here (W329);
/// pfilter must raise the SAME error, not silently score the wrong window.
#[test]
fn pfilter_unconditioned_wide_first_window_hard_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_wide_window_data(&data);
    let params = params_file(tmp.path());

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "an unconditioned wide first incidence window must be the W329 hard \
         error, not a silently mis-scored loglik:\nstderr={stderr}");
    assert!(stderr.contains("condition"),
        "the error must name the conditioning fix:\n{stderr}");
    assert!(stderr.contains("first window"),
        "the error must describe the wide first window:\n{stderr}");
}

/// `--condition-from "first_obs - 7 days"`: the warm-up [0, 35) is simulated
/// but not scored; the run succeeds with a finite loglik and says so.
#[test]
fn pfilter_condition_from_flag_applies_warmup() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_wide_window_data(&data);
    let params = params_file(tmp.path());

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--condition-from", "first_obs - 7 days",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "conditioned pfilter must run:\nstderr={stderr}");
    assert!(stderr.contains("conditioning window"),
        "the applied warm-up must be visible on stderr:\n{stderr}");
    let ll = run_loglik(&out);
    assert!(ll.is_finite() && ll < 0.0,
        "expected finite negative loglik, got {ll}");
}

/// The `--fit` toml's `condition_from` is honoured (it was silently dropped),
/// and at the same seed it produces the byte-identical loglik to the CLI flag
/// form — one conditioning semantics, two spellings.
#[test]
fn pfilter_fit_toml_condition_from_matches_flag() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_wide_window_data(&data);
    let params = params_file(tmp.path());

    let toml = write_fit_toml(tmp.path(), &data, Some("first_obs - 7 days"));

    let out_toml = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--fit", &toml.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out_toml.stderr);
    assert!(out_toml.status.success(),
        "pfilter must honour the --fit toml's condition_from:\nstderr={stderr}");
    assert!(stderr.contains("conditioning window"),
        "the toml-declared warm-up must be visible on stderr:\n{stderr}");
    let ll_toml = run_loglik(&out_toml);

    let out_flag = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &long_obs_model().to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--condition-from", "first_obs - 7 days",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let ll_flag = run_loglik(&out_flag);
    assert_eq!(ll_toml, ll_flag,
        "toml condition_from and --condition-from must be ONE semantics: \
         same seed, byte-identical loglik");
}

/// `profile` shares the gap and the fix: the same unconditioned wide first
/// window must raise the same W329 hard error before any profiling starts.
/// Invocation is the documented profile flow: `--fit` toml carries the fixed
/// values and the focal parameter's estimate spec; the sweep pins it per cell.
#[test]
fn profile_unconditioned_wide_first_window_hard_errors() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_wide_window_data(&data);
    let toml = write_fit_toml(tmp.path(), &data, None);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            "--fit", &toml.to_string_lossy(),
            "--sweep", "gamma=lin(0.05,0.2,2)",
            "--iterations", "1", "--starts", "1", "--suppress-warnings",
            "--rw-sd", "auto",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn profile");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "an unconditioned wide first incidence window must be the W329 hard \
         error in profile too:\nstderr={stderr}");
    assert!(stderr.contains("condition"),
        "the error must name the conditioning fix:\n{stderr}");
}

/// `profile --condition-from` runs the conditioned filter; the toml carries
/// no condition_from, so this also pins CLI-flag precedence on the profile
/// surface.
#[test]
fn profile_condition_from_flag_applies_warmup() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("cases_long.tsv");
    write_wide_window_data(&data);
    let toml = write_fit_toml(tmp.path(), &data, None);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "profile", &long_obs_model().to_string_lossy(),
            "--fit", &toml.to_string_lossy(),
            "--condition-from", "first_obs - 7 days",
            "--sweep", "gamma=lin(0.05,0.2,2)",
            "--iterations", "1", "--starts", "1", "--suppress-warnings",
            "--rw-sd", "auto",
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn profile");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "conditioned profile must run:\nstderr={stderr}");
    assert!(stderr.contains("conditioning window"),
        "the applied warm-up must be visible on stderr:\n{stderr}");
}
