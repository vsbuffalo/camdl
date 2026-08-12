//! gh#589: observations must not be quantized to the trajectory recording grid.
//!
//! Observations are projected from the RECORDED trajectory, so an emit time
//! falling between snapshots silently reads the earlier snapshot: a flow reports
//! its whole interval on one boundary and zeros elsewhere, a stock steps. The
//! emitted file still carries the requested timestamps, so the corruption is
//! invisible — and `--obs` output is normally fitted.
//!
//! Incident: `docs/dev/incidents/2026-08-12-obs-quantized-to-output-grid.md`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// Daily `emit_schedule`. The recording cadence is varied by the caller.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  daily_cases {
    columns       { time : time, daily_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    daily_cases   ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 21 'days }
"#;

fn setup(dir: &Path) -> (PathBuf, PathBuf) {
    let m = dir.join("q.camdl");
    std::fs::write(&m, MODEL).unwrap();
    let p = dir.join("p.toml");
    std::fs::write(&p, "beta = 0.5\ngamma = 0.1\n").unwrap();
    (m, p)
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

#[test]
fn misaligned_emit_schedule_is_refused_and_writes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path());
    let obs = tmp.path().join("obs.tsv");

    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        // Weekly recording against a daily emit schedule: days 1-6 would each
        // read the day-0 snapshot, so the week's incidence would land entirely
        // on day 7 and the rest would read zero.
        "--output-every", "7",
        "--obs", obs.to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a misaligned emit schedule must be refused, got exit 0.\nstderr={stderr}"
    );
    assert!(
        stderr.contains("not a recorded output time"),
        "the error must say which emit time is off the grid, got:\n{stderr}"
    );
    // Refused BEFORE emitting: a corrupt file that looks authoritative is worse
    // than no file.
    assert!(!obs.exists(), "no observation file may be written when refused");
}

#[test]
fn aligned_and_coarser_emit_schedules_are_accepted() {
    // Aligned: daily emit, default (fine) recording.
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = setup(tmp.path());
    let obs = tmp.path().join("obs.tsv");
    let out = run(&[
        "simulate", model.to_str().unwrap(),
        "--params", params.to_str().unwrap(),
        "--seed", "3",
        "--obs", obs.to_str().unwrap(),
        "--output-dir", tmp.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "an aligned emit schedule must still work:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The daily series must be real per-day incidence, not one lump per week.
    // This is the property the guard protects; it is asserted here so the test
    // fails if a future change reinstates the quantization without tripping the
    // guard.
    let txt = std::fs::read_to_string(&obs).unwrap();
    let vals: Vec<f64> = txt
        .lines()
        .skip(1)
        .filter_map(|l| l.split('\t').nth(1)?.parse().ok())
        .collect();
    assert!(vals.len() > 7, "expected a daily series, got {} rows", vals.len());
    let nonzero = vals.iter().filter(|v| **v > 0.0).count();
    assert!(
        nonzero > vals.len() / 2,
        "a daily incidence series should be mostly non-zero; {nonzero}/{} were. \
         Six-zeros-then-a-lump is the gh#589 signature.",
        vals.len()
    );

    // Coarser observations than recording is the SAFE direction — weekly emit
    // times land exactly on daily snapshots. This is the configuration the
    // recovery suite relies on, so it must keep working.
    let tmp2 = tempfile::tempdir().unwrap();
    let (m2, p2) = setup(tmp2.path());
    let weekly = MODEL.replace("every 1 'days", "every 7 'days");
    std::fs::write(&m2, weekly).unwrap();
    let obs2 = tmp2.path().join("obs.tsv");
    let out2 = run(&[
        "simulate", m2.to_str().unwrap(),
        "--params", p2.to_str().unwrap(),
        "--seed", "3",
        "--obs", obs2.to_str().unwrap(),
        "--output-dir", tmp2.path().join("r").to_str().unwrap(),
    ]);
    assert!(
        out2.status.success(),
        "observations coarser than recording must be accepted:\nstderr={}",
        String::from_utf8_lossy(&out2.stderr)
    );
}
