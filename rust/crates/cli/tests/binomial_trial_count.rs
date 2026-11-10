//! A binomial trial count `n` is checked against the observed counts when the
//! data is loaded, whenever `n` is decidable from the file alone, and `camdl
//! survey` reads the same `n = <column>` the fit does.
//!
//!   * `a_literal_n_below_a_count_is_refused_by_pfilter` — `n = 50` against a
//!     count of 90 was bound and scored `-inf` (printed as the loglik, exit 0);
//!     it is now refused with the row, the time and both values.
//!   * `an_expression_n_below_a_count_is_refused_by_pfilter` — the same for
//!     `n = tested * 2`.
//!   * `a_fractional_expression_n_is_refused_by_pfilter` — `n = tested / 2`
//!     with an odd `tested` would silently score `round(n)` trials.
//!   * `survey_scores_a_column_n_and_refuses_a_count_above_it` — survey bound
//!     the stream with no aux columns, so every `n = tested` stream failed with
//!     "aux column missing"; it now loads the column, scores, and refuses a
//!     transposed row like the fit loader does.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_binom_n_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// An SIR model whose survey stream scores `pos ~ binomial(n = <n_expr>, ...)`.
/// `tested` is declared as a data column only when `n_expr` reads it (a declared
/// column nothing reads is a compile error).
fn compile_model(dir: &Path, n_expr: &str) -> PathBuf {
    let tested_col = if n_expr.contains("tested") { ", tested : count" } else { "" };
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  survey {{
    columns       {{ time : time, pos : count{tested_col} }}
    projected  = I
    emit_schedule = every 1 'days
    pos ~ binomial(n = {n_expr}, p = projected / N0)
  }}
}}
scenarios {{ baseline {{ set = {{ beta = 0.3  gamma = 0.1  N0 = 1000 }} }} }}
init {{ S = 999  I = 1 }}
simulate {{ from = 0 'days  to = 6 'days }}
"#);
    let model = dir.join("m.camdl");
    std::fs::write(&model, src).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

fn camdl(dir: &Path) -> Command {
    let mut c = Command::new(camdl_bin());
    c.env_remove("CLICOLOR_FORCE").env("CAMDL_SKIP_VERSION_CHECK", "1").current_dir(dir);
    c
}

fn pfilter(dir: &Path, ir: &Path, data: &Path) -> Output {
    camdl(dir)
        .args(["pfilter", &ir.to_string_lossy(), "--data", &data.to_string_lossy(),
               "--scenario", "baseline", "--particles", "50", "--seed", "1"])
        .output().unwrap()
}

fn stderr(o: &Output) -> String { String::from_utf8_lossy(&o.stderr).into_owned() }

#[test]
fn a_literal_n_below_a_count_is_refused_by_pfilter() {
    let tmp = tempdir("literal");
    let ir = compile_model(&tmp.0, "50");
    // Control: every count at or below n runs and scores a finite loglik.
    let ok = write(&tmp.0, "ok.tsv", "time\tpos\n1\t1\n2\t3\n3\t40\n4\t4\n5\t2\n6\t1\n");
    let out = pfilter(&tmp.0, &ir, &ok);
    assert!(out.status.success(), "control must run: {}", stderr(&out));
    let loglik = String::from_utf8_lossy(&out.stdout).lines().find_map(|l| l.trim().parse::<f64>().ok());
    assert!(loglik.is_some_and(f64::is_finite), "control loglik must be finite: {:?}", out.stdout);

    let bad = write(&tmp.0, "bad.tsv", "time\tpos\n1\t1\n2\t3\n3\t90\n4\t4\n5\t2\n6\t1\n");
    let out = pfilter(&tmp.0, &ir, &bad);
    let err = stderr(&out);
    assert!(!out.status.success(), "a count above a literal n must not run:\n{err}");
    assert!(err.contains("row 2 (t = 3)") && err.contains("90") && err.contains("n = 50"),
        "the refusal must give the row, the time and both values:\n{err}");
}

#[test]
fn an_expression_n_below_a_count_is_refused_by_pfilter() {
    let tmp = tempdir("expr");
    let ir = compile_model(&tmp.0, "tested * 2");
    let bad = write(&tmp.0, "bad.tsv",
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t130\t50\n4\t4\t110\n5\t2\t100\n6\t1\t95\n");
    let out = pfilter(&tmp.0, &ir, &bad);
    let err = stderr(&out);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("row 2 (t = 3)") && err.contains("130") && err.contains("n = 100"), "{err}");
}

#[test]
fn a_fractional_expression_n_is_refused_by_pfilter() {
    let tmp = tempdir("fract");
    let ir = compile_model(&tmp.0, "tested / 2");
    let odd = write(&tmp.0, "odd.tsv",
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t5\t91\n4\t4\t110\n5\t2\t100\n6\t1\t96\n");
    let out = pfilter(&tmp.0, &ir, &odd);
    let err = stderr(&out);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("n = 45.5 is not an integer"), "{err}");
}

fn survey(dir: &Path, ir: &Path, data: &Path) -> Output {
    camdl(dir)
        .args(["survey", &ir.to_string_lossy(), "--data", &data.to_string_lossy(),
               "--estimate", "beta=0.1:0.6", "--estimate", "gamma=0.05:0.3",
               "--fixed", "N0=1000", "--n-points", "6", "--eval", "pfilter",
               "--eval-particles", "50", "--eval-replicates", "2", "--seed", "1"])
        .output().unwrap()
}

#[test]
fn survey_scores_a_column_n_and_refuses_a_count_above_it() {
    let tmp = tempdir("survey");
    let ir = compile_model(&tmp.0, "tested");
    let ok = write(&tmp.0, "ok.tsv",
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n");
    let out = survey(&tmp.0, &ir, &ok);
    assert!(out.status.success(), "survey on `n = tested` must run:\n{}", stderr(&out));

    let bad = write(&tmp.0, "bad.tsv",
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t90\t50\n4\t4\t110\n5\t2\t100\n6\t1\t95\n");
    let out = survey(&tmp.0, &ir, &bad);
    let err = stderr(&out);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("row 2 (t = 3)") && err.contains("exceeds denominator 'tested' = 50"), "{err}");
}
