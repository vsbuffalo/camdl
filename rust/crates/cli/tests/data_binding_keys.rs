//! gh#604 — a `[data.observations]` key that names no observation source must
//! be diagnosed as a BINDING error, before anything reads a byte of data.
//!
//! The motivating case is TOML scoping. `condition_from` is a top-level
//! fit.toml key; written below the `[data.observations]` header it becomes
//! `data.observations.condition_from` and binds an observation stream named
//! `condition_from` to the "path" `first_obs - 1 week`. The conditioning
//! window it was meant to set is then unset.
//!
//! The guard that catches this (`fit::runner::check_bound_sources`) existed but
//! was unreachable on the `fit run` path: the fit-level identity digests open
//! every bound path first, so the run died on `cannot read data file
//! 'condition_from'` — a missing-file diagnosis for a binding fault, pointing
//! the user at the filesystem instead of at the header their key sits under.
//!
//! Gates three properties:
//!   1. the misplaced top-level key reports as a binding error naming the
//!      declared sources, and says the key must sit above the first `[table]`;
//!   2. a mistyped stream name whose value IS a readable file gets the binding
//!      error WITHOUT the TOML-scoping hint (different fault, different fix);
//!   3. a well-formed binding to a missing file still reports as a missing
//!      file — the guard keys on the NAME, and does not swallow real I/O
//!      errors.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The release binary; the gate runner builds `--release` first.
fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

/// `seed_timing_dated` declares a single observation stream whose source is
/// `cases`, which is the only name a `[data.observations]` key may take here.
fn model_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../sim/tests/fixtures/seed_timing_dated.ir.json")
}

fn write_data(dir: &Path) -> PathBuf {
    let body = "time\tcases\n\
        2020-03-15\t3\n2020-03-16\t6\n2020-03-17\t11\n2020-03-18\t18\n\
        2020-03-19\t27\n2020-03-20\t31\n2020-03-21\t28\n2020-03-22\t20\n\
        2020-03-23\t13\n2020-03-24\t8\n";
    let p = dir.join("cases.tsv");
    std::fs::write(&p, body).unwrap();
    p
}

/// A one-stage fit.toml whose `[data.observations]` body is the variable under
/// test. Everything else is a working config.
fn write_fit_toml(dir: &Path, observations: &str) -> PathBuf {
    let body = format!(
        r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
{observations}

[estimate]
beta = {{ bounds = [0.1, 2.0], start = 0.6 }}
tau  = {{ bounds = [0.0, 60.0], start = 20.0 }}

[fixed]
gamma = 0.2
lambda = 2.0
w = 3.0
N0 = 1000
rho = 0.6
k = 10.0

[stages.posterior]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 3
cooling = 0.7
"#,
        out = dir.join("out").display(),
        ir = model_ir().display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, body).unwrap();
    p
}

fn run_fit(fit_toml: &Path) -> String {
    let out = Command::new(bin())
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run");
    assert!(!out.status.success(), "expected a non-zero exit; got success");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// (1) `condition_from` scoped into `[data.observations]` by its header.
#[test]
fn misplaced_top_level_key_reports_as_a_binding_error() {
    let tmp = tempfile::tempdir().unwrap();
    let data = write_data(tmp.path());
    let fit = write_fit_toml(
        tmp.path(),
        &format!(
            "cases = \"{}\"\ncondition_from = \"first_obs - 1 day\"",
            data.display()
        ),
    );

    let stderr = run_fit(&fit);

    // The fault is named, and named as a binding fault in the table the user
    // typed it in — not as a missing file.
    assert!(
        stderr.contains("[data.observations]") && stderr.contains("'condition_from'"),
        "must name the offending table and key; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("cannot read data file"),
        "must not diagnose a binding fault as a missing data file; got:\n{stderr}"
    );
    // The cause (TOML scoping) and the fix (move it above the first table).
    assert!(
        stderr.contains("TOP-LEVEL") && stderr.contains("ABOVE the first"),
        "must say the key belongs above the first [table]; got:\n{stderr}"
    );
    // The declared source, so the user can see what a real key looks like.
    assert!(
        stderr.contains("Available sources: cases"),
        "must list the declared sources; got:\n{stderr}"
    );
}

/// (2) A mistyped stream name bound to a file that DOES exist. Same binding
/// error, but the TOML-scoping hint would be wrong here, so it must be absent.
#[test]
fn mistyped_stream_name_omits_the_toml_scoping_hint() {
    let tmp = tempfile::tempdir().unwrap();
    let data = write_data(tmp.path());
    let fit = write_fit_toml(tmp.path(), &format!("casez = \"{}\"", data.display()));

    let stderr = run_fit(&fit);

    assert!(
        stderr.contains("'casez' is not an observation source")
            && stderr.contains("Available sources: cases"),
        "must name the key and the declared sources; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("TOP-LEVEL"),
        "the value is a readable file, so the misplaced-top-level-key hint is \
         the wrong diagnosis and must not fire; got:\n{stderr}"
    );
}

/// (3) Negative control: a correctly-named binding to a file that is not there
/// must still report the missing file. The guard keys on the name; it must not
/// widen into a file-existence check and swallow real I/O errors.
#[test]
fn correct_key_with_a_missing_file_still_reports_the_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("not_written.tsv");
    let fit = write_fit_toml(tmp.path(), &format!("cases = \"{}\"", missing.display()));

    let stderr = run_fit(&fit);

    assert!(
        stderr.contains("cannot read data file") && stderr.contains("cases"),
        "a real missing file must still surface as a missing file; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("is not an observation source"),
        "'cases' IS the declared source — the binding guard must not fire; got:\n{stderr}"
    );
}
