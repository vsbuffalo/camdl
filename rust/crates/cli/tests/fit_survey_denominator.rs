//! Stage 2: per-observation survey denominators (`binomial(n = tested)`)
//! through the `camdl fit` / `camdl pfilter` paths.
//!
//! A survey-positivity stream scores `pos ~ binomial(n = tested, p = ...)` where
//! `tested` is a DECLARED AUX VALUE COLUMN bound per observation from the data
//! file (NOT a parameter / cell shape — there is no `Counted` variant). The aux
//! value is read by name through `Expr::ObsColumnRef`. (2026-06-10 observation
//! data-entry §3, §6.1.)
//!
//! Coverage:
//!   * `fit_survey_positivity_runs_and_is_finite` — a binomial-positivity stream
//!     with a per-row `tested` denominator loads, binds the aux column, and
//!     produces a finite loglik. The headline end-to-end Stage-2 path.
//!   * `value_exceeds_denominator_is_a_located_data_error` — a transposed row
//!     (`pos > tested`) is rejected at bind with a row number — a DATA error
//!     caught at load, not a fit-time `-Inf` (`value ≤ n`, §3.2).
//!   * `missing_denominator_is_a_hole_not_nan` — a row whose `tested` is `NA`
//!     becomes a hole (no term scored; `binomial(n = NaN)` is unconstructible),
//!     so the fit still runs finite. Present-together-or-hole (§6.1).
//!
//! Silent-skip if the release binary / camdlc is not built (mirrors
//! `fit_sparse_holes`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        p.display()
    );
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
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
        "camdl_fit_survey_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// SIR with a prevalence-observed binomial-positivity survey stream. The survey
/// observes `pos` positives out of `tested` (a per-observation data column),
/// with positivity `p = I / N0` (the true prevalence). Prevalence (not
/// incidence) → no origin-window concern; this is about the aux binding.
fn write_model(dir: &Path) -> PathBuf {
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
  survey {
    columns       { time : time, pos : count, tested : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    pos ~ binomial(n = tested, p = projected / N0)
  }
}
scenarios { baseline { set = { beta = 0.3  gamma = 0.1  N0 = 1000 } } }
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("survey.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("survey.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// A well-formed survey: each row has `pos <= tested`, all denominators > 0.
fn write_survey_data(dir: &Path) -> PathBuf {
    let p = dir.join("survey.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

/// A survey with a TRANSPOSED row at t=3 (`pos = 90 > tested = 5`): a data
/// error the `value <= n` bind check must catch with a located row number.
fn write_transposed_data(dir: &Path) -> PathBuf {
    let p = dir.join("survey_bad.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\t3\t120\n3\t90\t5\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

/// A survey with a MISSING denominator at t=2 (`tested = NA`): the row becomes a
/// hole (present-together-or-hole) — no term scored, not `binomial(n = NaN)`.
fn write_missing_denom_data(dir: &Path) -> PathBuf {
    let p = dir.join("survey_hole.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\t3\tNA\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

/// A survey with a ZERO denominator at t=2 (`pos = 0, tested = 0`): nobody was
/// examined that day. gh#812 — a well-defined, non-identifying observation, not
/// a malformed row. The fit must RUN and the row must contribute exactly zero.
fn write_zero_denom_data(dir: &Path) -> PathBuf {
    let p = dir.join("survey_zero.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\t0\t0\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

/// The same survey with that row written `NA` instead. A zero-effort row and a
/// missing row contribute identically, so the two fits must agree exactly.
fn write_zero_as_na_data(dir: &Path) -> PathBuf {
    let p = dir.join("survey_zero_na.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\tNA\tNA\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

/// A zero denominator carrying a POSITIVE count (`pos = 3, tested = 0`): still
/// an error. No trials cannot yield a success.
fn write_zero_denom_positive_count(dir: &Path) -> PathBuf {
    let p = dir.join("survey_zero_bad.tsv");
    std::fs::write(&p,
        "time\tpos\ttested\n1\t1\t100\n2\t3\t0\n3\t5\t90\n4\t4\t110\n5\t2\t100\n6\t1\t95\n").unwrap();
    p
}

fn write_if2_toml(dir: &Path, ir: &Path, data: &Path) -> PathBuf {
    let out_root = dir.join("results_if2");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
survey = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.15 }}
[fixed]
N0 = 1000
[stages.posterior]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 30
iterations = 5
cooling = 0.9
"#,
        out = out_root.display(), ir = ir.display(), data = data.display());
    let p = dir.join("fit_if2.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

fn run_fit(bin: &Path, fit_toml: &Path, seed: &str) -> std::process::Output {
    Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", seed])
        .output()
        .expect("camdl fit must invoke")
}

fn best_loglik(stdout: &str, stderr: &str) -> Option<f64> {
    let combined = format!("{stdout}\n{stderr}");
    combined.lines().find_map(|l| {
        let idx = l.find("loglik=")?;
        let rest = &l[idx + "loglik=".len()..];
        rest.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
            .find(|tok| !tok.is_empty())
            .and_then(|tok| tok.parse::<f64>().ok())
    })
}

#[test]
fn fit_survey_positivity_runs_and_is_finite() {
    let camdl = camdl_bin();
    let tmp = tempdir("ok");
    let ir = write_model(tmp.path());
    let data = write_survey_data(tmp.path());
    let toml = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(out.status.success(),
        "camdl fit on a binomial-positivity survey (n = tested aux column) must \
         load + bind the aux + run.\nstdout:\n{stdout}\nstderr:\n{stderr}");

    let ll = best_loglik(&stdout, &stderr)
        .unwrap_or_else(|| panic!("no `loglik=` line in fit output:\n{stdout}\n{stderr}"));
    assert!(ll.is_finite() && ll < 0.0,
        "survey fit must report a finite negative loglik (the binomial scored \
         against the per-row `tested` denominator), got {ll}");
}

#[test]
fn value_exceeds_denominator_is_a_located_data_error() {
    let camdl = camdl_bin();
    let tmp = tempdir("transposed");
    let ir = write_model(tmp.path());
    let data = write_transposed_data(tmp.path());
    let toml = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stderr = String::from_utf8_lossy(&out.stderr);

    // A transposed row (pos > tested) is a DATA error caught at bind, with a
    // row number — not a silent fit-time -Inf.
    assert!(!out.status.success(),
        "a survey row with pos > tested must be rejected at bind, not run:\n{stderr}");
    assert!(stderr.contains("exceeds denominator") || stderr.contains("value ≤ n")
            || (stderr.contains("denominator") && stderr.contains("tested")),
        "the bind error must name the value-vs-denominator violation:\n{stderr}");
    // Located: the t=3 row is the third data row (file line 4).
    assert!(stderr.contains("row 2") || stderr.contains("row 3") || stderr.contains("transposed"),
        "the bind error must locate the offending row:\n{stderr}");
}

#[test]
fn missing_denominator_is_a_hole_not_nan() {
    let camdl = camdl_bin();
    let tmp = tempdir("holedenom");
    let ir = write_model(tmp.path());
    let data = write_missing_denom_data(tmp.path());
    let toml = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // A missing `tested` makes the row a hole (no term), NOT `binomial(n = NaN)`
    // that would poison the loglik. The fit must still run finite.
    assert!(out.status.success(),
        "a survey row with a missing (NA) denominator must become a hole, not \
         error or NaN.\nstdout:\n{stdout}\nstderr:\n{stderr}");
    let ll = best_loglik(&stdout, &stderr)
        .unwrap_or_else(|| panic!("no `loglik=` line:\n{stdout}\n{stderr}"));
    assert!(ll.is_finite() && ll < 0.0,
        "survey fit with a missing denominator (hole) must stay finite, got {ll}");
}

/// gh#812: a zero denominator is a well-defined observation, and a fit carrying
/// one must run.
///
/// `n = 0` means nobody was examined that day — routine in surveillance data
/// (weekends, stockouts, a lab that did not run). With no trials there is
/// exactly one possible outcome, so the term is exactly zero for every
/// parameter value: non-identifying, not invalid. camdl's kernel already
/// computed this correctly; only a bind-time guard refused it.
#[test]
fn a_zero_denominator_runs_and_warns_rather_than_refusing() {
    let camdl = camdl_bin();
    let tmp = tempdir("zero_denom");
    let ir = write_model(tmp.path());
    let data = write_zero_denom_data(tmp.path());
    let toml = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(out.status.success(),
        "a zero denominator must not refuse the fit:\n{stderr}");
    assert!(best_loglik(&stdout, &stderr).is_some_and(|ll| ll.is_finite()),
        "the fit must reach a finite loglik:\n{stderr}");
    // Reported once, with the row, so a zero-effort row is not silently
    // confused with a missing one.
    assert!(stderr.contains("denominator") && stderr.contains("= 0"),
        "the zero-denominator rows must be reported:\n{stderr}");
    assert!(stderr.contains("NA"),
        "the warning must say what to write if the rows are MISSING rather \
         than zero-effort:\n{stderr}");
}

/// gh#812: a zero-effort row and a missing row contribute identically, so the
/// two fits must agree — bit-identically, which is why the kernel
/// short-circuits to a literal 0.0 rather than letting the general formula
/// reach 0 to lgamma round-off.
#[test]
fn a_zero_denominator_scores_exactly_as_a_hole_does() {
    let camdl = camdl_bin();
    let tmp = tempdir("zero_vs_na");
    let ir = write_model(tmp.path());

    let zero = write_if2_toml(tmp.path(), &ir, &write_zero_denom_data(tmp.path()));
    let na_dir = tmp.path().join("na");
    std::fs::create_dir_all(&na_dir).unwrap();
    let ir2 = write_model(&na_dir);
    let na = write_if2_toml(&na_dir, &ir2, &write_zero_as_na_data(&na_dir));

    let o1 = run_fit(&camdl, &zero, "1");
    let o2 = run_fit(&camdl, &na, "1");
    let ll1 = best_loglik(&String::from_utf8_lossy(&o1.stdout), &String::from_utf8_lossy(&o1.stderr));
    let ll2 = best_loglik(&String::from_utf8_lossy(&o2.stdout), &String::from_utf8_lossy(&o2.stderr));

    assert!(ll1.is_some() && ll2.is_some(), "both fits must produce a loglik");
    assert_eq!(ll1, ll2,
        "a zero-effort row and a missing row must score identically — got \
         {ll1:?} (n = 0) vs {ll2:?} (NA)");
}

/// gh#812 does not relax the half of the check that matters: a positive count
/// against zero trials is still impossible.
#[test]
fn a_positive_count_against_zero_trials_is_still_an_error() {
    let camdl = camdl_bin();
    let tmp = tempdir("zero_denom_bad");
    let ir = write_model(tmp.path());
    let data = write_zero_denom_positive_count(tmp.path());
    let toml = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "a positive count against a zero denominator must be rejected:\n{stderr}");
    assert!(stderr.contains("row 1") && stderr.contains("denominator"),
        "the error must locate the row and name the denominator:\n{stderr}");
}
