//! gh#670 — `camdl survey` must report the fraction of Latin-hypercube
//! points whose log-likelihood came back non-finite, split by cause.
//!
//! The fraction is the pre-run answer to "what share of my bounds box can
//! the model not produce this data from at all?" — the same number that
//! surfaced downstream only *after* a multi-hour fit refused half its
//! chains at initialisation.
//!
//! Every case below pins an **exactly known** infeasible count, not a
//! "something was printed" check. Latin-hypercube sampling stratifies each
//! dimension into `n_points` equal bins with one draw per bin
//! (`fit::init::build_lhs_chain_starts`), so a threshold placed on a bin
//! boundary of a linearly-mapped box splits the points exactly:
//!
//!   * `offset` over `[0, 1998]`, 10 points, projection `S - offset` with
//!     `S` pinned at 999 (`beta = 0`, so nothing leaves `S`): the 5 points
//!     with `offset >= 999` give a non-positive Poisson rate against a
//!     positive count, i.e. `-inf` from the **observation** term.
//!   * `beta` over `[0, 1]`, 10 points, infection rate `(beta - shift)`
//!     with `shift = 0.2`: the 2 points with `beta < 0.2` make the rate
//!     negative, which the chain-binomial propensity evaluator refuses —
//!     the model has no trajectory there, i.e. **model support**.
//!   * `offset` over `[0, 500]`: every point is feasible — the negative
//!     control, which must report 0.0% and must NOT emit the loud line.

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
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp {
    fn path(&self) -> &Path { &self.0 }
}
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gh670_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

fn compile(dir: &Path, stem: &str, src: &str) -> PathBuf {
    let model_path = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model_path, src).unwrap();
    let out = Command::new(camdlc_bin()).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir_path = dir.join(format!("{stem}.ir.json"));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Projection `S - offset` with the infection rate switched off by
/// `beta = 0`, so `S` is pinned at its initial 999 for the whole horizon
/// and the Poisson rate is the constant `999 - offset`. 999 is the
/// midpoint of the `[0, 1998]` box, i.e. exactly the boundary between the
/// 5th and 6th Latin-hypercube bin at 10 points: the upper five draws give
/// a non-positive rate against strictly positive counts, so the
/// observation term's support excludes the data there and nowhere else.
const OFFSET_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta   : rate  in [0.0, 1.0]
  shift  : rate  in [0.0, 1.0]
  offset : count in [0.0, 1998.0]
  gamma  : rate  in [0.01, 1.0]
  N0     : count in [100, 10000]
}
transitions {
  infection : S --> I @ (beta - shift) * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = S - offset
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

/// Infection rate `(beta - shift) * S * I / N0`: negative wherever
/// `beta < shift`, which the chain-binomial propensity evaluator rejects
/// (`SimError::NegativePropensity`) — the model cannot step there at all.
const SHIFT_MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.0, 1.0]
  shift : rate  in [0.0, 1.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ (beta - shift) * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

const CASES_TSV: &str = "time\tcases\n1\t10\n2\t11\n3\t10\n4\t12\n5\t11\n6\t10\n";

/// The parsed `feasibility` block of a survey's `summary.json`.
#[derive(Debug, PartialEq, Eq)]
struct Feasibility {
    n_points: u64,
    n_non_finite: u64,
    n_observation: u64,
    n_support: u64,
    n_filter_degenerate: u64,
}

fn read_feasibility(root: &Path) -> Feasibility {
    let summary = find_leaf_file(root, "summary.json");
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summary).unwrap()).unwrap();
    let f = v.get("feasibility").unwrap_or_else(|| panic!(
        "summary.json has no `feasibility` block (gh#670): {}",
        serde_json::to_string_pretty(&v).unwrap()));
    let get = |k: &str| -> u64 {
        f.get(k).and_then(|x| x.as_u64()).unwrap_or_else(|| panic!(
            "feasibility.{} missing or not an integer in {}", k, f))
    };
    Feasibility {
        n_points: get("n_points"),
        n_non_finite: get("n_non_finite"),
        n_observation: get("n_observation"),
        n_support: get("n_support"),
        n_filter_degenerate: get("n_filter_degenerate"),
    }
}

fn find_leaf_file(root: &Path, name: &str) -> PathBuf {
    let mut stack = vec![root.join("surveys")];
    while let Some(dir) = stack.pop() {
        if dir.join("landscape.tsv").is_file() {
            let f = dir.join(name);
            assert!(f.is_file(), "survey leaf {} has no {}", dir.display(), name);
            return f;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
            }
        }
    }
    panic!("no survey leaf under {}", root.display());
}

/// Run `camdl survey` and return its stderr.
fn run_survey(root: &Path, ir: &Path, data: &Path, args: &[&str]) -> String {
    let mut cmd = Command::new(camdl_bin());
    cmd.env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["survey", &ir.to_string_lossy()])
        .args(["--data", &data.to_string_lossy()])
        .args(args)
        .args(["--n-points", "10", "--seed", "1"])
        .args(["--output", &root.to_string_lossy()]);
    let out = cmd.output().expect("spawn camdl survey");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "camdl survey failed:\n{}", stderr);
    stderr
}

/// The exact loud-line marker. Matching the whole sentence (not a bare
/// "warning:" substring, of which `survey` prints several) is what makes
/// the negative control below discriminating.
const LOUD_MARKER: &str =
    "warning: most of this parameter box is infeasible";

// ── Case A: exactly 5 of 10 points infeasible from the observation term ──

#[test]
fn observation_support_excludes_data_over_half_the_box() {
    let tmp = tempdir("obs");
    let ir = compile(tmp.path(), "offset", OFFSET_MODEL);
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, CASES_TSV).unwrap();
    let root = tmp.path().join("results");

    let stderr = run_survey(&root, &ir, &data, &[
        "--estimate", "offset=0.0:1998.0",
        "--fixed", "beta=0.0", "--fixed", "shift=0.0",
        "--fixed", "gamma=0.1", "--fixed", "N0=1000",
        "--eval", "simulate",
    ]);

    // The summary line, verbatim: count, percentage, and the by-cause split.
    let expected = "survey: 5 of 10 points (50.0%) scored a non-finite \
                    log-likelihood - 5 observation term, 0 model support, \
                    0 filter degeneracy";
    assert!(stderr.contains(expected),
        "expected summary line\n  {expected}\nnot found in stderr:\n{stderr}");

    // Half the box infeasible is at the loud threshold — the line must fire.
    assert!(stderr.contains(LOUD_MARKER),
        "50% infeasible must emit the loud line; stderr:\n{stderr}");

    assert_eq!(read_feasibility(&root), Feasibility {
        n_points: 10, n_non_finite: 5,
        n_observation: 5, n_support: 0, n_filter_degenerate: 0,
    });
}

// ── Case B: exactly 2 of 10 points outside the model's own support ───────

#[test]
fn model_support_violated_below_the_loud_threshold() {
    let tmp = tempdir("support");
    let ir = compile(tmp.path(), "shift", SHIFT_MODEL);
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, CASES_TSV).unwrap();
    let root = tmp.path().join("results");

    let stderr = run_survey(&root, &ir, &data, &[
        "--estimate", "beta=0.0:1.0",
        "--fixed", "shift=0.2", "--fixed", "gamma=0.1", "--fixed", "N0=1000",
        "--eval", "pfilter", "--eval-particles", "200",
        "--eval-replicates", "1",
    ]);

    let expected = "survey: 2 of 10 points (20.0%) scored a non-finite \
                    log-likelihood - 0 observation term, 2 model support, \
                    0 filter degeneracy";
    assert!(stderr.contains(expected),
        "expected summary line\n  {expected}\nnot found in stderr:\n{stderr}");

    // A fifth of the box is a normal amount of infeasible corner for a
    // hyperrectangle over a non-rectangular feasible set — no loud line.
    assert!(!stderr.contains(LOUD_MARKER),
        "20% infeasible must NOT emit the loud line; stderr:\n{stderr}");

    assert_eq!(read_feasibility(&root), Feasibility {
        n_points: 10, n_non_finite: 2,
        n_observation: 0, n_support: 2, n_filter_degenerate: 0,
    });
}

// ── Case C: negative control — every point feasible ─────────────────────

#[test]
fn all_finite_survey_reports_zero_and_stays_quiet() {
    let tmp = tempdir("clean");
    let ir = compile(tmp.path(), "offset", OFFSET_MODEL);
    let data = tmp.path().join("cases.tsv");
    std::fs::write(&data, CASES_TSV).unwrap();
    let root = tmp.path().join("results");

    let stderr = run_survey(&root, &ir, &data, &[
        "--estimate", "offset=0.0:500.0",
        "--fixed", "beta=0.0", "--fixed", "shift=0.0",
        "--fixed", "gamma=0.1", "--fixed", "N0=1000",
        "--eval", "simulate",
    ]);

    // 0% is reported explicitly — "the whole box is feasible" is itself
    // the answer a modeller came for, not an absence of output.
    let expected = "survey: 0 of 10 points (0.0%) scored a non-finite log-likelihood";
    assert!(stderr.contains(expected),
        "expected summary line\n  {expected}\nnot found in stderr:\n{stderr}");
    // With nothing infeasible there is no cause to attribute.
    assert!(!stderr.contains("observation term"),
        "an all-finite survey must not print a by-cause split; stderr:\n{stderr}");
    assert!(!stderr.contains(LOUD_MARKER),
        "an all-finite survey must not emit the loud line; stderr:\n{stderr}");

    assert_eq!(read_feasibility(&root), Feasibility {
        n_points: 10, n_non_finite: 0,
        n_observation: 0, n_support: 0, n_filter_degenerate: 0,
    });
}
