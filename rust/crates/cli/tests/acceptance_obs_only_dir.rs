//! Acceptance test for finding #5 / minor-2 (CLI review): obs-only output
//! must support multi-cadence streams via an explicit directory flag, and the
//! single-file multi-cadence error must point the user at the dir options.
//!
//! Proposal: docs/dev/proposals/2026-05-28-simulate-batch-coherence-and-obs-ensembles.md
//!
//! Verified (at time of writing):
//!   - `--obs-only-dir` does NOT exist (`args/mod.rs` has obs/obs_dir/obs_only
//!     only), though run-spec §3.1.1 lists it (`ObsOutput::OnlyDir`).
//!   - The multi-cadence single-file error (`main.rs:608-619`) names only
//!     `--obs-dir`.
//!
//! Both assertions below are RED until Stage 3 lands `ObsOutput::OnlyDir`
//! and updates the error text.

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
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// SIR with TWO observation streams at different cadences (weekly + biweekly)
/// — the shape that cannot be honestly serialized into one wide TSV.
fn write_sir_two_cadence_obs(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R }

let N = S + I + R

parameters {
  beta  : rate        in [0.01, 5.0]
  gamma : rate        in [0.01, 5.0]
  rho   : probability in [0.001, 1.0]
  N0    : count       in [100, 1000000]
  I0    : count       in [1, 1000]
}

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
  R = 0
}

simulate { from = 0 'days  to = 56 'days }

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected  = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases ~ poisson(rate = rho * projected)
  }
  biweekly_recoveries {
    columns       { time : time, biweekly_recoveries : count }
    projected  = incidence(recovery)
    emit_schedule = every 14 'days
    biweekly_recoveries ~ poisson(rate = rho * projected)
  }
}
"#;
    std::fs::write(path, src).unwrap();
}

fn write_params(path: &Path) {
    std::fs::write(path,
        "beta = 0.6\ngamma = 0.2\nrho = 0.5\nN0 = 10000\nI0 = 10\n").unwrap();
}

/// #5(A) — `--obs-only-dir DIR` writes one TSV per stream and suppresses the
/// trajectory (ObsOutput::OnlyDir).
#[test]
fn obs_only_dir_writes_one_file_per_stream() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    let params = tmp.path().join("p.toml");
    let obs_dir = tmp.path().join("obs_out");
    write_sir_two_cadence_obs(&model);
    write_params(&params);

    let out = Command::new(&bin)
        .args(["simulate", &model.to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--backend", "chain_binomial",
               "--dt", "1",
               "--seed", "1",
               "--obs-only-dir", &obs_dir.to_string_lossy()])
        .output().expect("spawn");
    assert!(out.status.success(),
        "`--obs-only-dir` must be accepted and run (run-spec §3.1.1 \
         ObsOutput::OnlyDir). stderr:\n{}", String::from_utf8_lossy(&out.stderr));

    let mut streams: Vec<String> = std::fs::read_dir(&obs_dir)
        .unwrap_or_else(|e| panic!("obs dir {} not created: {}", obs_dir.display(), e))
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().map(|x| x == "tsv").unwrap_or(false))
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    streams.sort();
    assert_eq!(streams, vec!["biweekly_recoveries".to_string(), "weekly_cases".to_string()],
        "obs-only-dir must write exactly one TSV per stream");

    // OnlyDir suppresses trajectory: stdout must not carry a trajectory.
    assert!(out.stdout.is_empty() || !String::from_utf8_lossy(&out.stdout).contains("\tS\t"),
        "--obs-only-dir must suppress trajectory output");
}

/// #5(B) — the multi-cadence single-file error must name BOTH `--obs-dir` and
/// `--obs-only-dir` so the user can discover the directory escape hatch
/// regardless of which obs mode they were reaching for.
#[test]
fn multi_cadence_single_file_error_names_both_dir_flags() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    let params = tmp.path().join("p.toml");
    write_sir_two_cadence_obs(&model);
    write_params(&params);

    let out = Command::new(&bin)
        .args(["simulate", &model.to_string_lossy(),
               "--params", &params.to_string_lossy(),
               "--backend", "chain_binomial",
               "--dt", "1",
               "--seed", "1",
               "--obs", &tmp.path().join("single.tsv").to_string_lossy()])
        .output().expect("spawn");
    assert!(!out.status.success(),
        "two streams with different cadences cannot share one wide TSV — \
         `--obs` must reject this.");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--obs-dir"),
        "multi-cadence error should name --obs-dir: {}", stderr);
    assert!(stderr.contains("--obs-only-dir"),
        "multi-cadence error should ALSO name --obs-only-dir so the \
         obs-only user finds the escape hatch: {}", stderr);
}
