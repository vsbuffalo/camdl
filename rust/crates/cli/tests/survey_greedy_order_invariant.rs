//! `camdl survey` evaluates LHS points in a greedy-nearest order (drill toward
//! the running-best point) so the best-loglik metric converges fast and the
//! interesting region fills in first. That reordering must be *purely* a
//! scheduling change: every point is still evaluated, and each point's result
//! is keyed on `(seed, point_id)`, so `landscape.tsv` must be **byte-identical**
//! regardless of the order points happen to run in.
//!
//! This test pins that invariant the only way it can be observed from outside:
//! run the same survey twice with different `--parallel` throttles. A different
//! thread budget means a different batch width, hence a different greedy
//! evaluation order — and yet the two landscape files must match exactly. If a
//! future change makes a point's result depend on when it ran (a shared RNG
//! stream, an order-dependent accumulator), this test goes red.
//!
//! `--eval simulate` keeps it deterministic and sub-second (no particle filter).

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
        "camdl_survey_order_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A small deterministic SIR + dataset. Three estimated params (one fixed) so
/// the transform-normalized distance metric that drives the greedy order has a
/// nontrivial space to walk.
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

/// The one `surveys/…` leaf holding `landscape.tsv`.
fn find_landscape(root: &Path) -> PathBuf {
    let mut stack = vec![root.join("surveys")];
    while let Some(dir) = stack.pop() {
        if dir.join("landscape.tsv").is_file() {
            return dir.join("landscape.tsv");
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
            }
        }
    }
    panic!("no landscape.tsv under {}", root.display());
}

/// Run a survey with a given `--parallel` throttle; return the landscape bytes.
fn run_survey(bin: &Path, ir: &Path, data: &Path, out_root: &Path, parallel: u32) -> Vec<u8> {
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "survey", &ir.to_string_lossy(),
            "--data", &data.to_string_lossy(),
            "--estimate", "beta=0.001:5.0",
            "--estimate", "gamma=0.01:1.0",
            "--fixed", "N0=1000",
            "--eval", "simulate",
            "--n-points", "250",
            "--seed", "1",
            "--parallel", &parallel.to_string(),
            "--output", &out_root.to_string_lossy(),
        ])
        .output().expect("spawn camdl survey");
    assert!(out.status.success(),
        "camdl survey (--parallel {}) must exit 0.\nstderr:\n{}",
        parallel, String::from_utf8_lossy(&out.stderr));
    std::fs::read(find_landscape(out_root)).expect("read landscape.tsv")
}

#[test]
fn survey_landscape_is_byte_identical_across_parallelism() {
    let bin = camdl_bin();
    let tmp = tempdir("inv");
    let (ir, data) = write_fixture(tmp.path());

    // Single-threaded (batch width 1) vs multi-threaded (wider batches) exercise
    // two very different greedy evaluation orders over the same 250 points.
    let serial   = run_survey(&bin, &ir, &data, &tmp.path().join("p1"), 1);
    let parallel = run_survey(&bin, &ir, &data, &tmp.path().join("p4"), 4);

    // Non-vacuous: header + 250 data rows.
    let n_lines = serial.iter().filter(|&&b| b == b'\n').count();
    assert!(n_lines >= 250,
        "expected ~250 landscape rows, got {} lines — fixture/eval may be broken", n_lines);

    assert!(serial == parallel,
        "survey landscape.tsv must be byte-identical regardless of evaluation \
         order (greedy scheduling is order-only); serial vs parallel differ.\n\
         serial {} bytes, parallel {} bytes", serial.len(), parallel.len());
}
