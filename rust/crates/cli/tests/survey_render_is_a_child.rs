//! `survey --render` writes `landscape.html` as a child of the survey's leaf
//! (`<leaf>/landscape_html/landscape.html`, declared in `run.json`'s
//! `children`), never one of the leaf's own files, as `simulate --event-log`
//! writes its log. What one run id holds is then the same whichever
//! invocation stored it, and rendering a stored survey leaves it reusable.
//!
//! Before, a fresh `--render` run added `landscape.html` to the leaf's
//! manifest, so the leaf's files depended on whether the first invocation
//! asked for it; and `--render` on a stored survey wrote the file into the
//! completed leaf unmanifested, after which the store read the leaf as stale
//! and the next identical survey computed it again.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.01, 1.0]
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
    projected     = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 6 'days }
"#;

fn bin(rel: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    assert!(p.exists(), "{} missing — run `make build`", p.display());
    p
}

fn survey(dir: &Path, extra: &[&str]) -> Output {
    let out = Command::new(bin("../../target/release/camdl"))
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["survey", "m.ir.json", "--data", "cases.tsv", "--estimate", "beta=0.1:0.9"])
        .args(["--fixed", "gamma=0.2", "--fixed", "N0=1000", "--n-points", "4", "--eval", "simulate"])
        .args(["--progress", "none"])
        .args(extra)
        .output()
        .unwrap();
    assert!(out.status.success(), "survey {extra:?}:\n{}", String::from_utf8_lossy(&out.stderr));
    out
}

/// The survey leaf under `dir`'s store and its `run.json`.
fn leaf(dir: &Path) -> (PathBuf, serde_json::Value) {
    let mut stack = vec![dir.join("results/surveys")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let v = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                return (p.parent().unwrap().to_path_buf(), v);
            }
        }
    }
    panic!("no survey leaf under {}", dir.display());
}

fn setup() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("m.camdl"), MODEL).unwrap();
    let out = Command::new(bin("../../../ocaml/_build/default/bin/camdlc.exe")).arg(dir.join("m.camdl")).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(dir.join("m.ir.json"), &out.stdout).unwrap();
    std::fs::write(dir.join("cases.tsv"), "time\tcases\n1\t12\n2\t14\n3\t17\n4\t19\n5\t22\n6\t24\n").unwrap();
    tmp
}

#[test]
fn the_rendered_landscape_is_a_child_of_the_leaf() {
    let (plain, rendered) = (setup(), setup());
    survey(plain.path(), &[]);
    survey(rendered.path(), &["--render"]);
    let ((_, a), (dir, b)) = (leaf(plain.path()), leaf(rendered.path()));
    assert_eq!(a["run_id"], b["run_id"], "--render does not change the run id");
    let files = |v: &serde_json::Value| -> Vec<String> { v["artifacts"].as_object().unwrap().keys().cloned().collect() };
    assert_eq!(files(&a), files(&b), "the leaf's own files are the same with or without --render");
    assert!(!dir.join("landscape.html").exists(), "the HTML is not one of the leaf's files");
    assert!(dir.join("landscape_html/landscape.html").is_file(), "the HTML is in its child");
    assert_eq!(b["children"]["landscape_html"].as_array().map(Vec::len), Some(1), "the child is declared:\n{b:#}");
}

#[test]
fn rendering_a_stored_survey_leaves_it_reusable() {
    let tmp = setup();
    let dir = tmp.path();
    survey(dir, &[]);
    let o = survey(dir, &["--render"]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("cached"), "the stored survey is served");
    let (leaf_dir, rec) = leaf(dir);
    assert!(leaf_dir.join("landscape_html/landscape.html").is_file(), "the HTML is written into its child");
    assert_eq!(rec["children"]["landscape_html"].as_array().map(Vec::len), Some(1), "and declared:\n{rec:#}");
    let o = survey(dir, &[]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("cached") && !err.contains("stored "), "the survey is still served from the store:\n{err}");
}

/// The rendered landscape is made from the leaf alone: the same survey run
/// from a copy of the model under another file name is the stored survey,
/// and rendering it finds the child already holding its HTML, rather than
/// reporting different bytes under one run id as an identity defect (the
/// page's title was the model file's name). A second render says so.
#[test]
fn the_rendered_landscape_depends_on_the_leaf_alone() {
    let tmp = setup();
    let dir = tmp.path();
    std::fs::copy(dir.join("m.ir.json"), dir.join("other.ir.json")).unwrap();
    survey(dir, &["--render"]);
    let o = survey(dir, &["--render"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("already rendered"), "a second render writes nothing and says so:\n{err}");
    let o = Command::new(bin("../../target/release/camdl"))
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["survey", "other.ir.json", "--data", "cases.tsv", "--estimate", "beta=0.1:0.9"])
        .args(["--fixed", "gamma=0.2", "--fixed", "N0=1000", "--n-points", "4", "--eval", "simulate"])
        .args(["--progress", "none", "--render"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("cached"), "the same survey under another model file name is the stored one:\n{err}");
    assert!(!err.contains("identity defect") && !err.contains("HTML render failed"), "no defect reported:\n{err}");
    assert!(err.contains("already rendered"), "{err}");
}
