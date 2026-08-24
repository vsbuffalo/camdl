//! A `camdl profile` run's storage folder (its content-addressed *base* dir)
//! includes the sweep grid, so a **distinct grid is a distinct run**. Before
//! this, the base was keyed on the model + swept-param *names* but not the
//! sweep *values*, so re-running the same params over a different range merged
//! the two grids' cells into one base dir — a jagged union when the cell
//! coordinates didn't line up (garki report).
//!
//! This pins the fix from outside: running two grids (same model + same swept
//! params, different ranges) into the same output root produces two distinct
//! base dirs, while re-running the *same* grid reuses its base dir (stable —
//! still a cache hit within a grid).

use std::collections::BTreeSet;
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

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_profile_grid_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin();
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
scenarios {
  baseline { set = { beta = 0.3  gamma = 0.1  N0 = 1000 } }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    (ir_path, data_path)
}

/// Run a profile with the given sweep specs into `root` (`CAMDL_OUTPUT_DIR`).
fn run_profile(bin: &Path, ir: &Path, data: &Path, root: &Path, beta: &str, gamma: &str) {
    run_profile_with(bin, ir, data, root, beta, gamma, "auto");
}

fn run_profile_with(
    bin: &Path, ir: &Path, data: &Path, root: &Path,
    beta: &str, gamma: &str, rw_sd: &str,
) {
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--sweep", beta,
            "--sweep", gamma,
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "5",
            "--starts", "1",
            "--rw-sd", rw_sd,
            "--seed", "1",
        ])
        .output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile must exit 0.\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// The base dirs under `<root>/profiles/` (the `<stem>-<basehash>` folders).
fn base_dirs(root: &Path) -> BTreeSet<String> {
    let profiles = root.join("profiles");
    std::fs::read_dir(&profiles).map(|rd| rd.flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()).unwrap_or_default()
}

/// The perturbation magnitudes are part of the method, so they belong in the
/// key. `--rw-sd` is a REQUIRED flag whose values drive the IF2 sampler, yet
/// only the estimated *set* was keyed (incidentally, via the priors blob) —
/// the magnitudes were not. So `--rw-sd "beta=0.05"` after `--rw-sd
/// "beta=0.5"` was served the FIRST run's profile, cell for cell, under a
/// "cached — resuming" line (2026-08-23 audit).
#[test]
fn distinct_rw_sd_gets_a_distinct_base_dir() {
    let bin = camdl_bin();
    let tmp = tempdir("rwsd");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("out");
    let (beta, gamma) = ("beta=lin(0.1,0.5,3)", "gamma=lin(0.03,0.25,3)");

    // rw_sd rides the `stage` level (the method config), not the base dir —
    // so the observable is the set of stage segments under a point. The
    // magnitude must be on a NON-focal parameter: beta and gamma are pinned
    // by the sweep, so only N0 is actually estimated here.
    run_profile_with(&bin, &ir, &data, &root, beta, gamma, "N0=5.0");
    let after_a = stage_dirs(&root);
    assert_eq!(after_a.len(), 1, "one run → one stage segment, got {:?}", after_a);

    // Same magnitudes again — a cache hit, no new segment (the negative
    // control: this must NOT fork, or the assertion below proves nothing).
    run_profile_with(&bin, &ir, &data, &root, beta, gamma, "N0=5.0");
    assert_eq!(stage_dirs(&root), after_a,
        "re-running with the SAME rw_sd must reuse its stage segment");

    // Different magnitudes — a different sampler, so a different artifact.
    run_profile_with(&bin, &ir, &data, &root, beta, gamma, "N0=50.0");
    let after_b = stage_dirs(&root);
    assert_eq!(after_b.len(), 2,
        "a different --rw-sd must create a distinct stage segment rather than \
         being served the previous run's landscape — got {:?}", after_b);
    assert!(after_b.is_superset(&after_a),
        "the first run's stage segment must still be present alongside the second's");
}

/// The `stage` segments (`<method>-<methodhash>`) across every point, which is
/// where the method config — particles, iterations, cooling, rw_sd, init —
/// lands in the leaf path.
fn stage_dirs(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut stack = vec![root.join("profiles")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() { continue }
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("if2-") || name.starts_with("pmmh-") {
                out.insert(name);
            } else {
                stack.push(p);
            }
        }
    }
    out
}

#[test]
fn distinct_grids_get_distinct_base_dirs() {
    let bin = camdl_bin();
    let tmp = tempdir("id");
    let (ir, data) = write_fixture(tmp.path());
    let root = tmp.path().join("out");

    // Grid A.
    run_profile(&bin, &ir, &data, &root, "beta=lin(0.1,0.5,3)", "gamma=lin(0.03,0.25,3)");
    let after_a = base_dirs(&root);
    assert_eq!(after_a.len(), 1, "one grid → one base dir, got {:?}", after_a);

    // Grid A again — same grid must reuse the same base dir (still cache-stable).
    run_profile(&bin, &ir, &data, &root, "beta=lin(0.1,0.5,3)", "gamma=lin(0.03,0.25,3)");
    assert_eq!(base_dirs(&root), after_a,
        "re-running the SAME grid must reuse its base dir, not fork a new one");

    // Grid B — same model + same swept params, DIFFERENT ranges. Must be its
    // own base dir, not a merge onto grid A (the garki bug).
    run_profile(&bin, &ir, &data, &root, "beta=lin(0.2,0.6,3)", "gamma=lin(0.05,0.3,3)");
    let after_b = base_dirs(&root);
    assert_eq!(after_b.len(), 2,
        "a DIFFERENT grid must create a distinct base dir, not merge onto the \
         previous one — got {:?}", after_b);
    assert!(after_b.is_superset(&after_a),
        "grid A's base dir must still be present alongside grid B's");
}
