//! gh#764 + fix 3 of the PMMH proposal-adaptation proposal.
//!
//! Two properties of a finished PMMH stage, both of which fail against the
//! preflight this replaces:
//!
//!  1. The measured likelihood noise is **in the stage artifact**. It was
//!     computed at preflight, printed to stderr and dropped, so a downstream
//!     team asked why their chains would not move could not tell us their
//!     noise level without re-running an expensive fit. `fit_state.toml` now
//!     carries `[pf_noise]` with the spread of a single `log L̂` (`sigma`),
//!     the spread of the difference that enters the Metropolis ratio (`s`),
//!     and the particle and pair counts they were measured at — `sigma` scales
//!     as `1/√N`, so a spread without its particle count is unreadable.
//!
//!  2. The preflight **computes the acceptance ceiling** `2·Φ(-s/2)` and says
//!     which side of this run's own target `0.234 + 0.206/d` it sits on. The
//!     check it replaces printed a green `PF variance OK (target: 1-3)` for
//!     any spread in `[0.5, 5.0]`, a range mostly past the point where the
//!     Robbins-Monro scale adaptation loses its root.
//!
//! Skipped when the release binary / camdlc is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gh764_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A small SIR with a daily case series — enough observation windows for the
/// filter to have a spread worth measuring.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
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
simulate { from = 0 'days  to = 20 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    let mut data = String::from("time\tcases\n");
    let cases = [
        2, 3, 5, 7, 10, 12, 15, 18, 20, 22,
        21, 19, 17, 15, 13, 11, 9, 7, 6, 5,
    ];
    for (i, c) in cases.iter().enumerate() {
        data.push_str(&format!("{}\t{}\n", i + 1, c));
    }
    std::fs::write(&data_path, &data).unwrap();
    (ir_path, data_path)
}

/// `rho_line` is either empty (plain PMMH) or a `rho = ...` stage key.
fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, rho_line: &str) -> PathBuf {
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.001, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 1.0 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0],  prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
[stages.post]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 1
particles  = 2000
iterations = 40
burn_in    = 10
thin       = 1
init       = "single"
{rho}
"#,
        out  = dir.join("results").display(),
        ir   = ir.display(),
        data = data.display(),
        rho  = rho_line,
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    p
}

/// Read `key = value` out of the `[pf_noise]` table of a `fit_state.toml`.
///
/// Parsed by hand rather than through the `cli` types on purpose: the point of
/// the issue is that the value is readable from a finished run directory by
/// someone who is not running our code.
fn pf_noise_field(state_toml: &str, key: &str) -> Option<f64> {
    state_toml.lines()
        .skip_while(|l| l.trim() != "[pf_noise]")
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k.trim() == key).then(|| v.trim().parse::<f64>().ok())?
        })
}

/// The one `fit_state.toml` the stage wrote, under its content-addressed run
/// directory (`results/fits/fit-<hash>/01-<stage>-<hash>/seed_N-<hash>/`).
fn stage_state(results: &Path) -> String {
    let mut found = Vec::new();
    let mut stack = vec![results.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().and_then(|n| n.to_str()) == Some("fit_state.toml") {
                    found.push(p);
                }
            }
        }
    }
    assert_eq!(found.len(), 1, "expected exactly one fit_state.toml; got {found:?}");
    std::fs::read_to_string(&found[0]).expect("read fit_state.toml")
}

fn run_fit(fit_toml: &Path) -> std::process::Output {
    let out = Command::new(camdl_bin())
        .args(["fit", "run"]).arg(fit_toml)
        .args(["--seed", "1"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(),
        "pmmh `fit run` must succeed (exit 0):\n{}", String::from_utf8_lossy(&out.stderr));
    out
}

#[test]
fn pmmh_persists_the_measured_noise_and_reports_the_acceptance_ceiling() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc_bin().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("plain");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, "");
    let out = run_fit(&fit_toml);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    // ── (1) gh#764: the spread is in the artifact, not only on stderr ──
    let state = stage_state(&tmp.path().join("results"));
    assert!(state.contains("[pf_noise]"),
        "fit_state.toml must carry the measured likelihood noise (gh#764):\n{state}");

    let sigma = pf_noise_field(&state, "sigma")
        .expect("[pf_noise] must carry `sigma`, the spread of one log L-hat");
    let s = pf_noise_field(&state, "s")
        .expect("[pf_noise] must carry `s`, the spread of the Metropolis log-ratio");
    let n_particles = pf_noise_field(&state, "n_particles")
        .expect("[pf_noise] must carry the particle count — sigma scales as 1/sqrt(N), \
                 so a spread without it cannot be read");
    let pairs = pf_noise_field(&state, "pairs")
        .expect("[pf_noise] must carry the replicate count — it sets the standard error \
                 of both spreads");

    assert!(sigma.is_finite() && sigma > 0.0, "sigma = {sigma}");
    assert!(s.is_finite() && s > 0.0, "s = {s}");
    assert_eq!(n_particles, 2000.0, "particle count must be the one the stage ran");
    assert_eq!(pairs, 20.0, "pair count must be the one the preflight measured");
    // Plain PMMH re-draws all of its randomness between the two evaluations,
    // so their difference is at least as spread as either one of them. It is
    // *not* sigma*sqrt(2) in general: the second evaluation sits at a theta' a
    // full initial-proposal step away, where the filter has its own noise
    // level, and s is sqrt(sigma_theta^2 + sigma_theta'^2).
    assert!(s > sigma,
        "independent evaluations: s = {s} cannot be below sigma = {sigma}");

    // ── (2) Fix 3: the ceiling is computed and sided, not banded ──
    assert!(stderr.contains("acceptance ceiling"),
        "the preflight must report the acceptance ceiling:\n{stderr}");
    // The printed ceiling is 2*Phi(-s/2) of the s that was stored — the report
    // and the artifact are the same measurement, so a reader who has only the
    // run directory can recompute what the run was told.
    let expected_ceiling = 2.0 * sim::inference::normal_cdf(-s / 2.0);
    assert!(stderr.contains(&format!("= {:.1}%", expected_ceiling * 100.0)),
        "the printed ceiling must be 2*Phi(-s/2) for the stored s = {s} \
         (expected {:.1}%):\n{stderr}",
        expected_ceiling * 100.0);
    assert!(stderr.contains("target 0.234 + 0.206/d"),
        "the ceiling must be reported against this run's own target:\n{stderr}");
    assert!(stderr.contains("(d = 2)"),
        "the target must be stated at this run's own dimension:\n{stderr}");
    assert!(!stderr.contains("PF variance OK"),
        "a hand-tuned band must not bless the run:\n{stderr}");
}

#[test]
fn correlated_pmmh_measures_and_persists_under_its_own_scheme() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc_bin().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    // The same fixture and the same seed, twice: the stages differ only in
    // whether `rho` is set. That is what makes the comparison below a test of
    // the measurement rather than of the label.
    let plain_tmp = tempdir("cpm_plain");
    let (ir, data) = write_fixture(plain_tmp.path());
    run_fit(&write_fit_toml(plain_tmp.path(), &ir, &data, ""));
    let plain_sigma = pf_noise_field(&stage_state(&plain_tmp.path().join("results")), "sigma")
        .expect("plain sigma");

    let tmp = tempdir("cpm");
    let (ir, data) = write_fixture(tmp.path());
    let fit_toml = write_fit_toml(tmp.path(), &ir, &data, "rho = 0.95");
    let out = run_fit(&fit_toml);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();

    let state = stage_state(&tmp.path().join("results"));
    let sigma = pf_noise_field(&state, "sigma").expect("sigma");
    let s = pf_noise_field(&state, "s").expect("s");
    let rho = pf_noise_field(&state, "rho").expect(
        "[pf_noise] must record the scheme `s` was measured under — `s` is a \
         property of the scheme, not only of the model");

    assert_eq!(rho, 0.95);
    assert!(sigma.is_finite() && sigma > 0.0, "sigma = {sigma}");
    assert!(s.is_finite() && s > 0.0, "s = {s}");
    // The evaluations went through the correlated filter with pre-drawn
    // randoms, not the plain bootstrap filter. If the preflight had called the
    // plain filter regardless of `rho` — which is the defect fix 3 names — the
    // two stages would have drawn the same seeds at the same base theta and
    // reported a bit-identical sigma.
    assert_ne!(sigma, plain_sigma,
        "the correlated stage must be measured under its own scheme; sigma \
         identical to the plain stage's ({plain_sigma}) means the plain \
         bootstrap filter was used for both");
    assert!(stderr.contains("correlated evaluations (rho = 0.95)"),
        "the report must name the scheme it measured under:\n{stderr}");
}
