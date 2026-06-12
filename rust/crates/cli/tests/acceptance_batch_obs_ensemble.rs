//! Acceptance test for finding #4 (CLI review): an *ensemble* of synthetic
//! observations must land in the CAS so a posterior-predictive / fan-chart
//! view on the observable is possible.
//!
//! Proposal: docs/dev/proposals/2026-05-28-simulate-batch-coherence-and-obs-ensembles.md
//!
//! Verified gap (at time of writing): `--cas` is single-run only
//! (`main.rs:503-513`) and `batch run` never samples observations
//! (`batch.rs:589` writes only `traj.tsv`; `BatchArgs` has no obs fields).
//! You can get an ensemble of trajectories into the CAS but never an ensemble
//! of observations.
//!
//! The CAS obs location is already designed in `cas/mod.rs:11-25`:
//!   seed_{n}/obs/{obs_hash}-{obs_seed}/<stream>.tsv
//! This test pins that contract: after `batch run` with obs enabled over
//! several seeds, every seed leaf carries an `obs/` subtree with at least one
//! stream TSV. RED until Stage 3 of the unification implements it.

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

/// SIR with a single daily reported-cases observation stream. `rho` (reporting
/// fraction) and all structural params are supplied via params.toml so the
/// model is fully determined for forward simulation + obs sampling.
fn write_sir_with_obs(path: &Path) {
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

simulate { from = 0 'days  to = 30 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    projected  = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}
"#;
    std::fs::write(path, src).unwrap();
}

/// CAS sim leaves: `sims/<sim>/<scen>/seed_N/` dirs containing run.json.
/// Every dir containing a `run.json`, at any depth — the factored CAS path
/// is 5 levels deep, and the obs ensemble is a declared `obs/` child below it.
fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
            }
        }
    }
    out
}

/// Find `*.tsv` files under a `seed_N/obs/` subtree (any obs_hash-obs_seed dir).
fn obs_stream_files(seed_leaf: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let obs_root = seed_leaf.join("obs");
    let Ok(dirs) = std::fs::read_dir(&obs_root) else { return out; };
    for d in dirs.flatten() {
        if !d.path().is_dir() { continue; }
        let Ok(files) = std::fs::read_dir(d.path()) else { continue; };
        for f in files.flatten() {
            let p = f.path();
            if p.extension().map(|e| e == "tsv").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out
}

/// #4 — obs ensemble in the CAS. `batch run` over 3 seeds with obs enabled
/// must deposit, under every seed leaf, an `obs/<...>/<stream>.tsv`.
#[test]
fn batch_writes_observation_ensemble_into_cas() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    let params = tmp.path().join("params.toml");
    let output = tmp.path().join("output");
    write_sir_with_obs(&model);
    std::fs::write(&params,
        "beta = 0.6\ngamma = 0.2\nrho = 0.5\nN0 = 10000\nI0 = 10\n").unwrap();

    // [obs] enabled = true is the minimal first-cut surface from the proposal.
    let batch = tmp.path().join("batch.toml");
    std::fs::write(&batch, format!(r#"
[config]
model = "{model}"
params = "{params}"
output_dir = "{out}"
backend = "chain_binomial"
dt = 1
seeds = {{ list = [1, 2, 3] }}
parallel = 1

[[scenario]]
name = "baseline"

[obs]
enabled = true
"#, model = model.display(), params = params.display(), out = output.display())).unwrap();

    let run = Command::new(&bin)
        .args(["batch", "run", &batch.to_string_lossy()])
        .output().expect("spawn");
    assert!(run.status.success(),
        "batch run with [obs] enabled must succeed. stderr:\n{}",
        String::from_utf8_lossy(&run.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 3,
        "expected one run leaf per seed (3), got {:?}", leaves);

    for leaf in &leaves {
        let streams = obs_stream_files(leaf);
        assert!(!streams.is_empty(),
            "seed leaf {} must contain obs/<obs_hash-obs_seed>/<stream>.tsv \
             (the designed CAS obs layout, cas/mod.rs:11). Found none — \
             batch did not sample/write the observation ensemble.",
            leaf.display());

        // The `cases` stream file must have content beyond a header.
        let cases = streams.iter()
            .find(|p| p.file_stem().map(|s| s == "cases").unwrap_or(false))
            .unwrap_or_else(|| panic!(
                "expected a `cases.tsv` obs stream under {}; found {:?}",
                leaf.display(), streams));
        let body = std::fs::read_to_string(cases).unwrap();
        let data_lines = body.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .count();
        assert!(data_lines >= 2,
            "cases.tsv at {} should have a header + sampled rows, got {} lines",
            cases.display(), data_lines);
    }
}
