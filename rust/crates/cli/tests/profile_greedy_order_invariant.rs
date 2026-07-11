//! `camdl profile` evaluates its grid cells with a greedy priority work-queue:
//! N workers each pull the pending `(cell × seed × start)` job whose cell is
//! nearest the current best-loglik cell, so cores stay busy while evaluation
//! drills toward the optimum and the progress bar's best-loglik metric
//! converges fast. That scheduling must be *purely* an ordering change — every
//! job is still evaluated exactly once, and each writes its own
//! content-addressed leaf keyed on `(grid, seed, start)`, so the per-cell
//! `mle.toml` results must be **byte-identical** regardless of the order (and
//! thread count) the cells happen to run in.
//!
//! This pins the invariant the only way it's observable from outside: run the
//! same 2D profile twice with different `--parallel` throttles — a different
//! thread budget means a different work-queue order — and require the full set
//! of per-cell `mle.toml` leaves to match exactly. If a future change makes a
//! cell's result depend on when it ran, or drops/double-runs a cell, this goes
//! red.
//!
//! Small IF2 grid (3×2 cells, 30 particles × 5 iters) keeps it fast.

use std::collections::BTreeMap;
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
        "camdl_profile_order_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Tiny SIR fixture with a `baseline` scenario supplying base parameter values
/// (the profile pins the swept params at grid values and optimizes the rest).
/// Same shape as `profile_diagnostics.rs`.
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

/// Run a 2D profile with a given `--parallel` throttle, leaves under `root`
/// (`CAMDL_OUTPUT_DIR`); return every per-cell `mle.toml` keyed by its path
/// relative to `root` (the content-addressed leaf path, which is identical
/// across runs with identical inputs).
fn run_profile(bin: &Path, ir: &Path, data: &Path, root: &Path, parallel: u32)
    -> BTreeMap<String, String>
{
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_OUTPUT_DIR", root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--scenario", "baseline",
            "--data", &data.to_string_lossy(),
            "--obs", "cases",
            "--sweep", "beta=lin(0.2,0.4,3)",
            "--sweep", "gamma=lin(0.05,0.15,2)",
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "5",
            "--starts", "1",
            "--rw-sd", "auto",
            "--seed", "1",
            "--parallel", &parallel.to_string(),
        ])
        .output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile (--parallel {}) must exit 0.\nstderr:\n{}",
        parallel, String::from_utf8_lossy(&out.stderr));

    // Walk for every `mle.toml` leaf; key by path relative to `root`.
    let mut leaves = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|n| n.to_str()) == Some("mle.toml") {
                    let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
                    let body = std::fs::read_to_string(&p).expect("read mle.toml");
                    leaves.insert(rel, body);
                }
            }
        }
    }
    leaves
}

#[test]
fn profile_leaves_are_byte_identical_across_parallelism() {
    let bin = camdl_bin();
    let tmp = tempdir("inv");
    let (ir, data) = write_fixture(tmp.path());

    // Single-threaded vs multi-threaded walk two different work-queue orders
    // over the same 3×2 grid.
    let serial   = run_profile(&bin, &ir, &data, &tmp.path().join("r1"), 1);
    let parallel = run_profile(&bin, &ir, &data, &tmp.path().join("r2"), 3);

    // Non-vacuous: every cell evaluated, none dropped or double-run.
    assert_eq!(serial.len(), 6, "expected 6 profile cells (3×2), got {}", serial.len());
    assert_eq!(parallel.len(), 6, "expected 6 profile cells (3×2), got {}", parallel.len());

    // Identical leaf paths (same CAS identities) and identical mle.toml bodies.
    assert!(serial == parallel,
        "profile per-cell mle.toml leaves must be byte-identical regardless of \
         evaluation order (work-queue scheduling is order-only). \n\
         serial paths:   {:?}\nparallel paths: {:?}",
        serial.keys().collect::<Vec<_>>(),
        parallel.keys().collect::<Vec<_>>());
}
