//! Sparse/holes through the `camdl fit` path.
//!
//! `camdl fit` (IF2, PGAS, PMMH, ODE-MLE, and the post-fit PFilter stage) must
//! LOAD a holed series (a value-column `NA` = a missing observation) and route
//! the holes — as `None` cells — through the already hole-correct scoring seam
//! (`MultiStreamObsModel`). A hole contributes NO likelihood term, but its
//! observation time stays in the grid so the incidence accumulator still resets
//! at it (pomp `accumvars` fixed-bin semantics).
//!
//! Before the loader change, `camdl fit` rejected `NA` at load with the
//! dense-loader error ("missing value `NA` ... not supported on this path");
//! the tests below are RED against that code and GREEN once the fit runner
//! loads via the cells loader and threads the cells into `build_obs_model`.
//!
//! Coverage:
//!   * `fit_if2_on_holed_series_runs_and_is_finite` — `camdl fit` (IF2) on a
//!     series with an interior hole loads, runs, and produces a finite best
//!     loglik (was: hard error at load).
//!   * `fit_pfilter_stage_loglik_matches_standalone_pfilter_on_holes` — the
//!     fit's internal PFilter-stage loglik on the holed series at FIXED params
//!     equals `camdl pfilter` on the SAME holed data at the same
//!     params/seed/particles. Same machinery → exact match. Pins that the fit
//!     path routes holes through the same hole-correct code, not a parallel
//!     reimplementation.
//!   * `dense_fit_pfilter_stage_matches_standalone_pfilter` — dense parity: a
//!     no-hole series scores identically through the fit path and standalone
//!     pfilter (the cell loader's all-`Some` output is the dense path).
//!
//! Silent-skip if the release binary / camdlc is not built (mirrors
//! `incidence_t0` / `fit_priors`).

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
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(),
        "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_fit_holes_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A tiny SIR with a prevalence-observed `cases` stream (Poisson). Returns the
/// compiled IR path. Prevalence (not incidence) so no origin-window concern;
/// the hole test is about loading + scoring, not incidence-bin alignment.
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
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
scenarios { baseline { set = { beta = 0.3  gamma = 0.1  N0 = 1000 } } }
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
    ir_path
}

/// Holed series: an interior hole (`NA`) at t=2, present first obs.
fn write_holed_data(dir: &Path) -> PathBuf {
    let p = dir.join("cases_holed.tsv");
    std::fs::write(&p, "time\tcases\n1\t2\n2\tNA\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    p
}

/// Dense version of the same series (the t=2 hole filled).
fn write_dense_data(dir: &Path) -> PathBuf {
    let p = dir.join("cases_dense.tsv");
    std::fs::write(&p, "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    p
}

/// The third encoding: the t=2 row is ABSENT rather than `NA`. t=2 is then not
/// on this stream's axis at all, so the observation grid is
/// [1, 3, 4, 5, 6] — one two-substep window among one-substep windows.
fn write_absent_row_data(dir: &Path) -> PathBuf {
    let p = dir.join("cases_absent.tsv");
    std::fs::write(&p, "time\tcases\n1\t2\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    p
}

/// A correlated-PMMH (`rho` set) fit.toml against `data`. Tiny counts — the
/// question is whether the observation grid is admitted and the pre-drawn noise
/// indexes correctly, not whether the chain converges.
fn write_correlated_pmmh_toml(dir: &Path, ir: &Path, data: &Path, tag: &str) -> (PathBuf, PathBuf) {
    let out_root = dir.join(format!("results_cpm_{tag}"));
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 0.3, prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }} }}
[fixed]
gamma = 0.1
N0 = 1000
[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 60
iterations = 12
burn_in = 2
init = "single"
rho = 0.99
"#,
        out = out_root.display(), ir = ir.display(), data = data.display());
    let p = dir.join(format!("fit_cpm_{tag}.toml"));
    std::fs::write(&p, toml).unwrap();
    (p, out_root)
}

/// Write an IF2 fit.toml against `data`. Tiny particle / iteration counts so it
/// finishes fast; the assertion is "loads + finite", not convergence.
fn write_if2_toml(dir: &Path, ir: &Path, data: &Path) -> (PathBuf, PathBuf) {
    let out_root = dir.join("results_if2");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
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
    (p, out_root)
}

/// Write a PFilter-stage fit.toml at FIXED params (start = the truth so the
/// PF runs at the same params the standalone `camdl pfilter` is given). One
/// replicate, prequential off (a holed series + prequential is rejected;
/// here we just want the bare filter loglik).
fn write_pfilter_toml(dir: &Path, ir: &Path, data: &Path, tag: &str) -> (PathBuf, PathBuf) {
    let out_root = dir.join(format!("results_pf_{tag}"));
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], start = 0.1 }}
[fixed]
N0 = 1000
[stages.check]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 200
replicates = 1
record_prequential = false
"#,
        out = out_root.display(), ir = ir.display(), data = data.display());
    let p = dir.join(format!("fit_pf_{tag}.toml"));
    std::fs::write(&p, toml).unwrap();
    (p, out_root)
}

fn run_fit(bin: &Path, fit_toml: &Path, seed: &str) -> std::process::Output {
    Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", seed])
        .output()
        .expect("camdl fit must invoke")
}

/// Read the single replicate's loglik from the PFilter stage's `logliks.tsv`.
fn read_pfilter_stage_loglik(out_root: &Path) -> f64 {
    // .../fits/<fit>/01-check-<hash>/seed_<S>-<hash>/logliks.tsv
    let mut found = None;
    for entry in walk(out_root) {
        if entry.file_name().is_some_and(|n| n == "logliks.tsv") {
            found = Some(entry);
            break;
        }
    }
    let path = found.unwrap_or_else(|| panic!("no logliks.tsv under {}", out_root.display()));
    let body = std::fs::read_to_string(&path).unwrap();
    // header `replicate\tloglik`, then `1\t<ll>`
    let line = body.lines().nth(1)
        .unwrap_or_else(|| panic!("logliks.tsv has no data row:\n{body}"));
    line.split('\t').nth(1)
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("cannot parse loglik from '{line}'"))
}

/// Run `camdl pfilter` standalone at the fixed truth params and return its
/// stdout loglik.
fn run_standalone_pfilter(bin: &Path, ir: &Path, data: &Path, seed: &str) -> f64 {
    let out = Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", ir.to_str().unwrap(),
            "--particles", "200", "--dt", "1", "--seed", seed,
            "--data", data.to_str().unwrap(),
            "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
        ])
        .output()
        .expect("camdl pfilter must invoke");
    assert!(out.status.success(),
        "standalone pfilter failed: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout.lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no loglik in pfilter stdout:\n{stdout}"))
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); } else { out.push(p); }
            }
        }
    }
    out
}

#[test]
fn fit_if2_on_holed_series_runs_and_is_finite() {
    let camdl = camdl_bin();
    let tmp = tempdir("if2");
    let ir = write_model(tmp.path());
    let data = write_holed_data(tmp.path());
    let (toml, _out) = write_if2_toml(tmp.path(), &ir, &data);

    let out = run_fit(&camdl, &toml, "1");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Must NOT fail at load with the dense-loader's NA rejection.
    assert!(out.status.success(),
        "camdl fit (IF2) on a holed series must load + run, not error.\n\
         stdout:\n{stdout}\nstderr:\n{stderr}");
    assert!(
        !stderr.contains("not supported on this path") && !stderr.contains("only handled by"),
        "fit must not reject the hole with the dense-loader error:\n{stderr}");

    // The reported best loglik must be finite (a hole omits a term; it does NOT
    // turn the loglik into -inf/NaN). The line is e.g.
    // "best chain: 1 (loglik=-13.31 ± 0.02)" — parse the token after `loglik=`.
    let combined = format!("{stdout}\n{stderr}");
    let best_ll = combined.lines()
        .find_map(|l| {
            let idx = l.find("loglik=")?;
            let rest = &l[idx + "loglik=".len()..];
            // first numeric token (handles "-13.31 ± 0.02" / "-13.31)" etc.)
            rest.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
                .find(|tok| !tok.is_empty())
                .and_then(|tok| tok.parse::<f64>().ok())
        })
        .unwrap_or_else(|| panic!("no `loglik=` line in fit output:\n{combined}"));
    assert!(best_ll.is_finite() && best_ll < 0.0,
        "fit on holed series must report a finite negative loglik, got {best_ll}");
}

#[test]
fn fit_pfilter_stage_loglik_matches_standalone_pfilter_on_holes() {
    let camdl = camdl_bin();
    let tmp = tempdir("pfmatch");
    let ir = write_model(tmp.path());
    let data = write_holed_data(tmp.path());
    let (toml, out_root) = write_pfilter_toml(tmp.path(), &ir, &data, "holed");

    // Same seed both sides: the PFilter stage's replicate-0 PF seed is
    // `seed ^ 0 = seed`, the same seed standalone pfilter uses; same
    // params/particles/dt → the bootstrap filter is identical → exact match.
    let seed = "7";
    let out = run_fit(&camdl, &toml, seed);
    assert!(out.status.success(),
        "fit PFilter stage on holed series must run:\n{}",
        String::from_utf8_lossy(&out.stderr));

    let fit_ll = read_pfilter_stage_loglik(&out_root);
    let standalone_ll = run_standalone_pfilter(&camdl, &ir, &data, seed);

    assert!(fit_ll.is_finite() && standalone_ll.is_finite(),
        "both logliks must be finite: fit={fit_ll}, standalone={standalone_ll}");
    // Same machinery, same seed → identical to numerical write precision.
    assert!((fit_ll - standalone_ll).abs() < 1e-3,
        "fit PFilter-stage loglik on holed data ({fit_ll}) must match standalone \
         pfilter ({standalone_ll}) — both route holes through the same \
         hole-correct scoring seam. diff = {}", (fit_ll - standalone_ll).abs());
}

#[test]
fn dense_fit_pfilter_stage_matches_standalone_pfilter() {
    // Dense parity: a no-hole series scores identically through the fit path
    // (the cell loader emits all-`Some`) and standalone pfilter — so the cell
    // loader does not perturb the no-hole result.
    let camdl = camdl_bin();
    let tmp = tempdir("densepar");
    let ir = write_model(tmp.path());
    let data = write_dense_data(tmp.path());
    let (toml, out_root) = write_pfilter_toml(tmp.path(), &ir, &data, "dense");

    let seed = "7";
    let out = run_fit(&camdl, &toml, seed);
    assert!(out.status.success(),
        "dense fit PFilter stage must run:\n{}",
        String::from_utf8_lossy(&out.stderr));

    let fit_ll = read_pfilter_stage_loglik(&out_root);
    let standalone_ll = run_standalone_pfilter(&camdl, &ir, &data, seed);
    assert!((fit_ll - standalone_ll).abs() < 1e-3,
        "dense parity: fit PFilter-stage loglik ({fit_ll}) must equal standalone \
         pfilter ({standalone_ll}). diff = {}", (fit_ll - standalone_ll).abs());
}

/// Correlated PMMH on a series whose interior row is ABSENT.
///
/// The pre-drawn noise CPM reuses across MCMC iterations is one block per
/// observation window, so the block sizes have to follow the grid. An absent
/// row merges two windows into one of twice the substeps, and `NA` is not a
/// substitute: an `NA` row keeps its time on the axis and resets the incidence
/// accumulator there, an absent row does neither
/// (`long_form_absent_row_is_not_scheduled`, `cli/src/pfilter.rs`). A daily
/// reporting series with one day of no reporting therefore has to reach the
/// filter as an irregular grid or not at all.
///
/// The dense arm runs alongside it so a failure here is attributable to the
/// grid rather than to anything else in the correlated path.
#[test]
fn fit_correlated_pmmh_runs_on_an_absent_row_series() {
    let camdl = camdl_bin();
    let tmp = tempdir("cpm_absent");
    let ir = write_model(tmp.path());

    let dense = write_dense_data(tmp.path());
    let (dense_toml, _) = write_correlated_pmmh_toml(tmp.path(), &ir, &dense, "dense");
    let dense_out = run_fit(&camdl, &dense_toml, "3");
    assert!(
        dense_out.status.success(),
        "correlated PMMH on the dense series must run. stderr:\n{}",
        String::from_utf8_lossy(&dense_out.stderr)
    );

    let absent = write_absent_row_data(tmp.path());
    let (absent_toml, _) = write_correlated_pmmh_toml(tmp.path(), &ir, &absent, "absent");
    let absent_out = run_fit(&camdl, &absent_toml, "3");
    assert!(
        absent_out.status.success(),
        "correlated PMMH must run on a grid with one merged window \
         (observations at 1, 3, 4, 5, 6). stderr:\n{}",
        String::from_utf8_lossy(&absent_out.stderr)
    );
}
