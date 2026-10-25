//! Acceptance test for finding #4 (CLI review): an *ensemble* of synthetic
//! observations must land in the store so a posterior-predictive / fan-chart
//! view on the observable is possible.
//!
//! Proposal: docs/dev/proposals/2026-05-28-simulate-batch-coherence-and-obs-ensembles.md
//!
//! The store's obs location is designed in `cas/mod.rs:11-25`:
//!   seed_{n}/obs/{obs_hash}-{obs_seed}/<stream>.tsv
//! These tests pin that contract on the multi-cell `simulate` path: after a
//! `--seeds` run with an observation flag, every seed leaf carries an `obs/`
//! subtree with at least one stream TSV — and asking for observations over a
//! store whose trajectories were computed without them writes them, rather
//! than reporting every cell cached and skipping them.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
    covers        = closing_at(time, 1 'days)
    projected  = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}
"#;
    std::fs::write(path, src).unwrap();
}

/// Every dir containing a `run.json`, at any depth — the factored store path
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

/// `simulate sir.camdl --params params.toml --backend chain_binomial --dt 1
/// --seeds <seeds> --output-dir <output> <extra…>`.
fn simulate(bin: &Path, model: &Path, params: &Path, output: &Path, seeds: &str, extra: &[&str]) -> Output {
    let mut cmd = Command::new(bin);
    cmd.args(["simulate", &model.to_string_lossy(),
              "--params", &params.to_string_lossy(),
              "--backend", "chain_binomial", "--dt", "1",
              "--seeds", seeds,
              "--output-dir", &output.to_string_lossy()]);
    cmd.args(extra);
    cmd.output().expect("spawn")
}

/// Asking for observations over a store whose trajectories were already
/// computed WITHOUT them must still produce observations.
///
/// The obs subtree is a child, written only by `merge_cell`, so a cached
/// trajectory used to skip the cell and take its obs with it: a run with an
/// observation flag over an obs-free store reported every cell cached and
/// wrote no observations at all, silently. The user asked for a thing, got
/// nothing, and the output said "cached".
#[test]
fn requesting_obs_over_an_obs_free_store_still_writes_observations() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    let params = tmp.path().join("params.toml");
    let output = tmp.path().join("output");
    write_sir_with_obs(&model);
    std::fs::write(&params,
        "beta = 0.6\ngamma = 0.2\nrho = 0.5\nN0 = 10000\nI0 = 10\n").unwrap();

    // First: trajectories only, no obs.
    let first = simulate(&bin, &model, &params, &output, "1,2", &[]);
    assert!(first.status.success(), "obs-free run failed. stderr:\n{}",
        String::from_utf8_lossy(&first.stderr));
    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 2, "expected one leaf per seed, got {:?}", leaves);
    for leaf in &leaves {
        assert!(!leaf.join("obs").is_dir(),
            "the obs-free run must not have written an obs subtree");
    }

    // Now ask for observations. Every trajectory is a cache hit, so the cells
    // would have been skipped — and their observations never written.
    let synth = tmp.path().join("synth");
    let second = simulate(&bin, &model, &params, &output, "1,2",
        &["--obs-dir", &synth.to_string_lossy()]);
    assert!(second.status.success(), "obs run failed. stderr:\n{}",
        String::from_utf8_lossy(&second.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 2, "requesting obs must not fork new trajectory leaves");
    for leaf in &leaves {
        assert!(leaf.join("obs").is_dir(),
            "requesting observations over an obs-free store must WRITE them, \
             not report the cell cached and skip them: {}", leaf.display());
        assert!(!obs_stream_files(leaf).is_empty(),
            "the obs subtree must contain per-stream files: {}", leaf.display());
    }
}

/// #4 — obs ensemble in the store. A `--seeds` run over 3 seeds with an
/// observation flag must deposit, under every seed leaf, an
/// `obs/<...>/<stream>.tsv`.
#[test]
fn a_multi_seed_run_writes_the_observation_ensemble_into_the_store() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir.camdl");
    let params = tmp.path().join("params.toml");
    let output = tmp.path().join("output");
    write_sir_with_obs(&model);
    std::fs::write(&params,
        "beta = 0.6\ngamma = 0.2\nrho = 0.5\nN0 = 10000\nI0 = 10\n").unwrap();

    let synth = tmp.path().join("synth");
    let run = simulate(&bin, &model, &params, &output, "1,2,3",
        &["--obs-dir", &synth.to_string_lossy()]);
    assert!(run.status.success(),
        "a --seeds run with --obs-dir must succeed. stderr:\n{}",
        String::from_utf8_lossy(&run.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 3,
        "expected one run leaf per seed (3), got {:?}", leaves);

    for leaf in &leaves {
        let streams = obs_stream_files(leaf);
        assert!(!streams.is_empty(),
            "seed leaf {} must contain obs/<obs_hash-obs_seed>/<stream>.tsv \
             (the designed store obs layout, cas/mod.rs:11). Found none — \
             the observation ensemble was not sampled/written.",
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
