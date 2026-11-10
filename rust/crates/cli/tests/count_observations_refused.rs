//! A count family scores non-negative integers; a data cell that is not one is
//! refused at load, naming the file, the column, the line and the value.
//!
//! Before the loader checked, the likelihood code coerced the cell instead
//! (`y.round().max(0.0)`): a `-4` in a Poisson column scored exactly as an
//! observed `0`, and a `4.4` exactly as a `4`, with exit 0 and a finite
//! log-likelihood. Every observation path reads its cells through the same
//! two parsers in `pfilter.rs` (the by-name column parser and the long-form
//! row parser), so these tests drive them through the commands that matter —
//! `pfilter`, `fit run` and `survey` — for every count family, the binomial
//! trial count, the Bernoulli outcome, and a stratified long-form stream.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust`", p.display());
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
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir()
        .join(format!("camdl_count_refused_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// A closed SIR observed by prevalence (`projected = I`) through one stream
/// whose `columns { }` and `~` line are `columns` and `law`. Compiled to IR
/// with the in-tree compiler, so the test reads no installed `camdlc`.
fn compile(dir: &Path, columns: &str, law: &str) -> PathBuf {
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate        in [0.01, 5.0]
  gamma : rate        in [0.01, 1.0]
  rho   : probability in [0.01, 1.0]
  N0    : count       in [100, 10000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  obs {{
    columns       {{ {columns} }}
    projected     = I
    emit_schedule = every 1 'days
    {law}
  }}
}}
init {{ S = 990  I = 10 }}
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

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(camdl_bin())
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .output()
        .unwrap()
}

const PARAMS: [&str; 8] = [
    "--param", "beta=0.5", "--param", "gamma=0.2", "--param", "rho=0.5", "--param", "N0=1000",
];

fn pfilter(dir: &Path, ir: &Path, data: &Path) -> Output {
    let mut args = vec!["pfilter", ir.to_str().unwrap(), "--particles", "50"];
    let binding = format!("obs={}", data.display());
    args.extend(["--data", &binding]);
    args.extend(PARAMS);
    camdl(dir, &args)
}

/// The refusal names everything a modeller needs to find and fix the cell.
fn assert_refused(out: &Output, file: &str, column: &str, line: usize, value: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "a {value} in column '{column}' must be refused, not scored; exit 0 with:\n{}{err}",
        String::from_utf8_lossy(&out.stdout));
    for needle in [file.to_string(), format!("'{column}'"), format!("line {line}"),
                   format!("'{value}'")] {
        assert!(err.contains(&needle), "the refusal must name {needle}:\n{err}");
    }
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

const CASES_COLUMNS: &str = "time : time, cases : count";

/// Each scalar count family, with a negative and a non-integer cell. Line 3 of
/// the file is the row for t = 2.
#[test]
fn pfilter_refuses_a_non_count_in_every_count_family() {
    let laws = [
        ("poisson", "cases ~ poisson(rate = rho * projected)"),
        ("neg_binomial", "cases ~ neg_binomial(mean = rho * projected, r = 10)"),
        ("zero_inflated",
         "cases ~ zero_inflated(base = neg_binomial(mean = rho * projected, r = 10), pi = 0.1)"),
    ];
    for (family, law) in laws {
        let tmp = tempdir(family);
        let ir = compile(&tmp.0, CASES_COLUMNS, law);
        for (tag, bad) in [("neg", "-4"), ("frac", "4.4")] {
            let name = format!("{family}_{tag}.tsv");
            let data = write(&tmp.0, &name,
                &format!("time\tcases\n1\t3\n2\t{bad}\n3\t6\n4\t5\n5\t4\n6\t3\n"));
            assert_refused(&pfilter(&tmp.0, &ir, &data), &name, "cases", 3, bad);
        }
    }
}

/// The binomial outcome AND its trial count `n = tested`: both are counts, and
/// a negative trial count was clamped to zero trials the same way.
#[test]
fn pfilter_refuses_a_non_count_in_a_binomial_outcome_or_trial_count() {
    for (family, law) in [
        ("binomial", "positives ~ binomial(n = tested, p = projected / N0)"),
        ("beta_binomial",
         "positives ~ beta_binomial(n = tested, mean = projected / N0, concentration = 20)"),
    ] {
        let tmp = tempdir(family);
        let ir = compile(&tmp.0, "time : time, positives : count, tested : count", law);
        let rows = |pos: &str, tested: &str| format!(
            "time\tpositives\ttested\n1\t1\t50\n2\t{pos}\t{tested}\n3\t2\t50\n\
             4\t1\t50\n5\t1\t50\n6\t0\t50\n");
        let name = format!("{family}_k.tsv");
        let data = write(&tmp.0, &name, &rows("-1", "50"));
        assert_refused(&pfilter(&tmp.0, &ir, &data), &name, "positives", 3, "-1");
        let name = format!("{family}_n.tsv");
        let data = write(&tmp.0, &name, &rows("1", "-50"));
        assert_refused(&pfilter(&tmp.0, &ir, &data), &name, "tested", 3, "-50");
        let name = format!("{family}_n_frac.tsv");
        let data = write(&tmp.0, &name, &rows("1", "49.5"));
        assert_refused(&pfilter(&tmp.0, &ir, &data), &name, "tested", 3, "49.5");
    }
}

/// A Bernoulli outcome is 0 or 1. Anything else was thresholded at 0.5.
#[test]
fn pfilter_refuses_a_bernoulli_outcome_that_is_not_zero_or_one() {
    let tmp = tempdir("bernoulli");
    let ir = compile(&tmp.0, "time : time, detected : count",
        "detected ~ bernoulli(p = projected / N0)");
    for (tag, bad) in [("neg", "-1"), ("two", "2")] {
        let name = format!("bernoulli_{tag}.tsv");
        let data = write(&tmp.0, &name,
            &format!("time\tdetected\n1\t0\n2\t{bad}\n3\t1\n4\t0\n5\t1\n6\t0\n"));
        assert_refused(&pfilter(&tmp.0, &ir, &data), &name, "detected", 3, bad);
    }
}

/// A stratified stream reads its file through the long-form parser, a second
/// reader of the same cells. The model is the two-patch long-form golden.
#[test]
fn pfilter_refuses_a_negative_count_in_a_long_form_stream() {
    let tmp = tempdir("long");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ir = Path::new(&manifest).join("../../../ocaml/golden/sir_two_patch_long_obs.ir.json");
    let params = write(&tmp.0, "params.toml", "beta = 0.05\ngamma = 0.1\nrho = 0.6\nk = 5.0\n");
    let name = "long.tsv";
    let data = write(&tmp.0, name,
        "time\tpatch\tcases\n7\turban\t12\n7\trural\t3\n\
         14\turban\t18\n14\trural\t-6\n21\turban\t25\n21\trural\t9\n");
    let binding = format!("cases={}", data.display());
    let out = camdl(&tmp.0, &[
        "pfilter", ir.to_str().unwrap(), "--params", params.to_str().unwrap(),
        "--data", &binding, "--particles", "100", "--dt", "1",
    ]);
    assert_refused(&out, name, "cases", 5, "-6");
}

/// `fit run` and `survey` read the same cells through their own entry points.
#[test]
fn fit_run_and_survey_refuse_a_negative_poisson_count() {
    let tmp = tempdir("fit_survey");
    let ir = compile(&tmp.0, CASES_COLUMNS, "cases ~ poisson(rate = rho * projected)");
    let name = "neg.tsv";
    let data = write(&tmp.0, name, "time\tcases\n1\t3\n2\t-4\n3\t6\n4\t5\n5\t4\n6\t3\n");
    // Paths relative to the fit.toml, which sits beside the model and data.
    let toml = write(&tmp.0, "fit.toml", r#"
output_dir = "results"
[model]
camdl = "m.ir.json"
[data.observations]
obs = "neg.tsv"
[estimate]
beta  = { bounds = [0.01, 5.0], start = 0.5 }
gamma = { bounds = [0.01, 1.0], start = 0.2 }
[fixed]
rho = 0.5
N0  = 1000
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 2
cooling    = 0.9
"#);
    let fit = camdl(&tmp.0, &["fit", "run", toml.to_str().unwrap()]);
    assert_refused(&fit, name, "cases", 3, "-4");

    let binding = data.display().to_string();
    let survey = camdl(&tmp.0, &[
        "survey", ir.to_str().unwrap(), "--data", &binding,
        "--estimate", "beta=0.1:1.0", "--fixed", "gamma=0.2", "--fixed", "rho=0.5",
        "--fixed", "N0=1000", "--n-points", "4", "--eval", "simulate",
    ]);
    assert_refused(&survey, name, "cases", 3, "-4");
}
