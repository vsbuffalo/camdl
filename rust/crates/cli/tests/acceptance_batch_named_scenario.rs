//! Acceptance test for finding #3 (CLI review): `batch run` must resolve
//! named model scenarios the same way `simulate --scenario` does.
//!
//! Proposal: docs/dev/proposals/2026-05-28-simulate-batch-coherence-and-obs-ensembles.md
//!
//! Verified cause (at time of writing): `batch.rs` builds each run's SimRun
//! with `scenario_name: None` hardcoded, so `params_resolver` never consults
//! `model.presets`. `simulate --scenario X` does
//! (`params_resolver.rs:398-427`). The `[[scenario]].name` is therefore a
//! label, not a reference into the model — a model whose named scenario is
//! the *sole source* of a parameter runs under simulate but fails under batch.
//!
//! These tests are RED against the pre-unification binary and must go GREEN
//! once `batch run` resolves named presets. They are CLI-level: they shell
//! out to the built `camdl` binary and exercise the real compile→resolve→
//! simulate→CAS pipeline.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> Option<PathBuf> {
    let bin = binary();
    if !bin.exists() {
        eprintln!("skipping: camdl binary not built at {}", bin.display());
        return None;
    }
    Some(bin)
}

/// Pure-death model whose parameter `mu` has **no default and no
/// params.toml value** — it is supplied *only* by the named scenario's
/// `set { }` block. This is the exact shape that exposes #3: simulate
/// resolves `mu` from the scenario; pre-fix batch drops it.
fn write_scenario_only_param_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S }

parameters {
  mu : rate in [0.001, 10.0]
}

init { S = 1000 }

transitions {
  death : S -->   @ mu * S
}

simulate { from = 0 'days  to = 20 'days }

scenarios {
  baseline { set = { mu = 0.1 } }
  fast     { set = { mu = 0.5 } }
}
"#;
    std::fs::write(path, src).unwrap();
}

/// Collect every directory three levels under `root` that contains a
/// `run.json` (the CAS `sims/<sim>/<scen>/seed_N/` leaves).
/// Every dir containing a `run.json`, at any depth — the factored CAS path
/// is 5 levels deep (model/config/params/scenario/seed), not 3.
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

/// Numeric data rows of a TSV trajectory (drop comment/blank/header lines).
/// Used for the determinism comparison: same RNG, same draw order ⇒ identical
/// numeric trajectory.
fn data_rows(tsv: &str) -> Vec<String> {
    tsv.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter(|l| {
            // keep only lines whose first field parses as a float (data rows),
            // dropping the header row (`t\tS`).
            l.split('\t').next()
                .map(|f| f.parse::<f64>().is_ok())
                .unwrap_or(false)
        })
        .map(str::to_string)
        .collect()
}

/// #3(A) — reproduction. A model scenario that is the sole source of a
/// parameter must resolve under `batch run`, with NO params file supplying
/// that parameter. Pre-fix this fails:
///   `Validation("parameter 'mu' has no value; supply it via --params ...")`
#[test]
fn batch_resolves_named_scenario_as_sole_param_source() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("pd.camdl");
    let output = tmp.path().join("output");
    write_scenario_only_param_model(&model);

    // Deliberately NO `params = ...` line: `mu` must come from the
    // model's `baseline` scenario `set { mu = 0.1 }`.
    let batch = tmp.path().join("batch.toml");
    std::fs::write(&batch, format!(r#"
[config]
model = "{model}"
output_dir = "{out}"
seeds = {{ n = 1 }}
parallel = 1

[[scenario]]
name = "baseline"
"#, model = model.display(), out = output.display())).unwrap();

    let run = Command::new(&bin)
        .args(["batch", "run", &batch.to_string_lossy()])
        .output().expect("spawn");

    assert!(run.status.success(),
        "batch run must resolve the model's `baseline` scenario (which supplies \
         mu=0.1 via set{{}}). It currently drops the scenario and errors. \
         stderr:\n{}", String::from_utf8_lossy(&run.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 1, "expected exactly one run leaf, got {:?}", leaves);
    assert!(leaves[0].join("traj.tsv").exists(),
        "the resolved baseline run must write traj.tsv");
}

/// #3(B) — determinism / equivalence. With a stochastic backend at a fixed
/// seed, the trajectory produced via `simulate --scenario fast` and the
/// trajectory produced via `batch run` for the same named scenario must have
/// byte-identical numeric data rows. This pins that batch routes through the
/// same resolve+simulate path (same params, same RNG draw order) — proposal
/// acceptance criterion #1.
#[test]
fn batch_named_scenario_matches_simulate_trajectory() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("pd.camdl");
    write_scenario_only_param_model(&model);

    // simulate --scenario fast (chain_binomial, seed 7) → out.tsv
    let sim_out = tmp.path().join("sim.tsv");
    let sim = Command::new(&bin)
        .args(["simulate", &model.to_string_lossy(),
               "--scenario", "fast",
               "--backend", "chain_binomial",
               "--dt", "1",
               "--seed", "7",
               "-o", &sim_out.to_string_lossy()])
        .output().expect("spawn");
    assert!(sim.status.success(),
        "simulate --scenario fast failed: {}", String::from_utf8_lossy(&sim.stderr));
    let sim_rows = data_rows(&std::fs::read_to_string(&sim_out).unwrap());
    assert!(!sim_rows.is_empty(), "simulate produced no data rows");

    // batch run, same scenario + seed + backend → CAS traj.tsv
    let output = tmp.path().join("output");
    let batch = tmp.path().join("batch.toml");
    std::fs::write(&batch, format!(r#"
[config]
model = "{model}"
output_dir = "{out}"
backend = "chain_binomial"
dt = 1
seeds = {{ list = [7] }}
parallel = 1

[[scenario]]
name = "fast"
"#, model = model.display(), out = output.display())).unwrap();

    let run = Command::new(&bin)
        .args(["batch", "run", &batch.to_string_lossy()])
        .output().expect("spawn");
    assert!(run.status.success(),
        "batch run --scenario fast failed: {}", String::from_utf8_lossy(&run.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 1, "expected one run leaf, got {:?}", leaves);
    let batch_rows = data_rows(&std::fs::read_to_string(leaves[0].join("traj.tsv")).unwrap());

    assert_eq!(sim_rows, batch_rows,
        "batch-run trajectory must be byte-identical (numeric rows) to \
         `simulate --scenario fast` at the same seed/backend. A mismatch means \
         the two paths resolved different params or consumed the RNG in a \
         different order.");
}
