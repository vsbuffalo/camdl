//! §9.10 integration regression for the 2026-05-22 calendar-time feature:
//! an `instant`-kind estimand (the seed time `tau`) is reported as a
//! **calendar date** in `camdl fit summary` when the model declares an
//! `origin`.
//!
//! Model: `crates/sim/tests/fixtures/seed_timing_dated.ir.json` — the
//! seed-timing SIR with `origin = date("2020-02-24")` and `tau : instant`.
//! We run a tiny IF2 fit (it need not converge — this is a rendering
//! regression, not a statistical one), then assert that the `tau` row in
//! the summary carries an ISO date alongside its numeric estimate, in
//! both the JSON and text formats.
//!
//! Shells out to the built `camdl` binary; silent-skip when it is absent
//! so the suite stays runnable in rust-only CI and before a build (same
//! convention as `fit_experiment_management.rs` / `seed_timing_e2e.rs`).
//! `CAMDL_SKIP_VERSION_CHECK=1` avoids a stale globally installed
//! `camdlc` making the test flaky.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn dated_model_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../sim/tests/fixtures/seed_timing_dated.ir.json")
}

struct TempDir(PathBuf);
impl TempDir {
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "camdl_caltime_fit_{}_{}_{}",
        tag,
        std::process::id(),
        ns
    ));
    std::fs::create_dir_all(&p).unwrap();
    TempDir(p)
}

/// Synthetic *dated* daily case data. The time column carries ISO dates
/// (`origin = 2020-02-24`), exercising the dated loader; counts trace a
/// small rise-and-fall so the fit has something to chew on. Values are
/// arbitrary — only that the fit runs and `tau` is estimated matters.
fn write_dated_data(dir: &Path) -> PathBuf {
    let data = "time\tcases\n\
        2020-03-15\t3\n\
        2020-03-16\t6\n\
        2020-03-17\t11\n\
        2020-03-18\t18\n\
        2020-03-19\t27\n\
        2020-03-20\t31\n\
        2020-03-21\t28\n\
        2020-03-22\t20\n\
        2020-03-23\t13\n\
        2020-03-24\t8\n";
    let path = dir.join("cases_dated.tsv");
    std::fs::write(&path, data).unwrap();
    path
}

/// Tiny IF2 fit.toml estimating `tau` (the instant estimand). 2 chains,
/// 4 iterations, 300 particles — sub-second, structural not statistical.
///
/// Particle count is load-bearing for the test passing, not just for speed:
/// the gh#110 PF-degeneracy watchdog makes a chain whose effective sample
/// size collapses a hard `PFDegenerate` error, and IF2 aborts the whole
/// `fit run` when any chain errors. On this 10-point dataset a starved
/// filter degenerates at obs window 2 — empirically at ≤100 particles, but
/// not at ≥200 (it is particle starvation, not a structural impossibility).
/// 300 keeps a comfortable margin while still running sub-second. Do not
/// lower it to "speed up the test": that reintroduces the degeneracy.
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, output_dir: &Path) -> PathBuf {
    let fit_toml = dir.join("fit.toml");
    let body = format!(
        r#"
output_dir = "{out}"
# Early origin (~20 d before the daily data) to estimate the seed time; condition
# one cadence before the first datum so the first incidence bin is one day, not
# the whole pre-data gap (gh#134 / W329).
condition_from = "first_obs - 1 day"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.1, 2.0], start = 0.6 }}
tau  = {{ bounds = [0.0, 60.0], start = 20.0 }}

[fixed]
gamma  = 0.2
lambda = 2.0
w      = 3.0
N0     = 1000
rho    = 0.6
k      = 10.0

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 300
iterations = 4
cooling    = 0.7
"#,
        out = output_dir.display(),
        ir = ir.display(),
        data = data.display(),
    );
    std::fs::write(&fit_toml, body).unwrap();
    fit_toml
}

fn exec_fit_run(camdl: &Path, fit_toml: &Path, output_dir: &Path) -> PathBuf {
    let out = Command::new(camdl)
        .arg("fit")
        .arg("run")
        .arg(fit_toml)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit run must invoke");
    assert!(
        out.status.success(),
        "camdl fit run failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let fits = output_dir.join("fits");
    let entries: Vec<PathBuf> = std::fs::read_dir(&fits)
        .unwrap_or_else(|_| panic!("no fits/ dir under {}", output_dir.display()))
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(entries.len(), 1, "expected one fit dir, got {:?}", entries);
    entries.into_iter().next().unwrap()
}

fn exec_summary(camdl: &Path, fit_dir: &Path, format: &str) -> String {
    let out = Command::new(camdl)
        .arg("fit")
        .arg("summary")
        .arg(fit_dir)
        .arg("--format")
        .arg(format)
        .arg("--no-color")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl fit summary must invoke");
    assert!(
        out.status.success(),
        "camdl fit summary --format {} failed: stderr={}",
        format,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// §9.10: `tau` (instant-kind) is reported as a calendar date once the
/// model carries `origin`. Asserts the date appears in JSON
/// (`estimate_date` sibling field on the `tau` parameter) and text
/// (a parenthesised ISO date in the `tau` row), and that the
/// non-instant `beta` is NOT date-annotated.
#[test]
fn instant_estimand_renders_as_calendar_date() {
    let camdl = camdl_bin();
    let ir = dated_model_ir();
    if !ir.exists() {
        return; // committed fixture absent (shouldn't happen in-tree)
    }

    let tmp = tempdir("dated");
    let data = write_dated_data(tmp.path());
    let output_dir = tmp.path().join("out");
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, &output_dir);
    let fit_dir = exec_fit_run(&camdl, &fit_toml, &output_dir);

    // ── JSON: tau parameter carries an `estimate_date` sibling field ──
    let json: serde_json::Value =
        serde_json::from_str(&exec_summary(&camdl, &fit_dir, "json"))
            .expect("summary JSON must parse");
    let stages = json["stages"].as_array().expect("stages array");
    assert!(!stages.is_empty(), "expected at least one stage: {json}");

    let mut saw_tau_date = false;
    let mut saw_beta = false;
    for stage in stages {
        let Some(params) = stage["parameters"].as_array() else { continue };
        for p in params {
            match p["name"].as_str() {
                Some("tau") => {
                    let date = p
                        .get("estimate_date")
                        .and_then(|d| d.as_str())
                        .unwrap_or_else(|| panic!(
                            "tau must carry estimate_date when origin is set: {p}"
                        ));
                    // origin 2020-02-24; any in-bounds tau ∈ [0,60] maps to
                    // a 2020 spring date. Assert ISO shape + plausible year.
                    assert!(
                        date.starts_with("2020-"),
                        "tau date should anchor to the 2020 origin, got {date}"
                    );
                    assert_eq!(date.len(), 10, "ISO YYYY-MM-DD, got {date}");
                    saw_tau_date = true;
                }
                Some("beta") => {
                    // Non-instant param: must NOT be date-annotated.
                    assert!(
                        p.get("estimate_date").is_none(),
                        "beta is not an instant — must omit estimate_date: {p}"
                    );
                    saw_beta = true;
                }
                _ => {}
            }
        }
    }
    assert!(saw_tau_date, "no tau parameter row found in summary JSON: {json}");
    assert!(saw_beta, "no beta parameter row found in summary JSON: {json}");

    // ── Text: tau row carries a parenthesised ISO date ───────────────
    let text = exec_summary(&camdl, &fit_dir, "text");
    let tau_line = text
        .lines()
        .find(|l| l.trim_start().starts_with("tau"))
        .unwrap_or_else(|| panic!("no tau row in text summary:\n{text}"));
    assert!(
        tau_line.contains("(2020-"),
        "tau text row must carry a calendar date: {tau_line}"
    );
}
