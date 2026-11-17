//! End to end: `nl-lbfgs` on an ODE model with scheduled events (gh#275 §1g).
//!
//! An age-structured SEIR whose only effects are yearly school-entry transfers
//! (a fraction of every young compartment moves to its old counterpart, four
//! transfers at one instant, mid-step at `dt = 1`) plus an importation — the
//! shape of the report that motivated the feature, which `nl-lbfgs` refused
//! for "scheduled interventions or events". The two estimated parameters, the
//! transfer fraction and the importation size, enter the likelihood ONLY
//! through the event jumps: the pure §1g case, every bit of their gradient is
//! the jump term.
//!
//!   (1) `nl-lbfgs` runs, and converges to the same optimum as the
//!       derivative-free `nl-sbplx` from the same start (to tolerance), in far
//!       fewer objective evaluations — the counts are printed;
//!   (2) the same model with the importation written as a fixed-count transfer
//!       is refused at config validation, before any chain runs or any leaf is
//!       written, naming the event and the construct — and the refusal is not
//!       dressed as "no finite log-likelihood ... starting values".
//!
//! The numerical correctness of the gradient is proved by the FD oracles
//! (`sim::inference::ode_grad::tests`, `sim/tests/event_jump_sensitivity.rs`);
//! this is the CLI-surface cell (config → gate → fit → outputs). Skipped when
//! the release binaries are missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("../../target/release/camdl")
}
fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = PathBuf::from(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!("camdl_lbfgs_events_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

const MODEL: &str = r#"
time_unit = 'days

compartments { S, E, I, R }

dimensions {
  age = [young, old]
}

stratify(by = age)

let N_local[a in age] = S[a] + E[a] + I[a] + R[a]

parameters {
  beta  : rate  in [0.005, 2.0]
  sigma : rate  in [0.05, 1.0]
  gamma : rate  in [0.05, 1.0]
  seed  : count in [0.5, 200.0]
  frac_age : probability in [0.02, 0.8]
}

tables {
  C_age : age × age = [[10.0, 3.0], [3.0, 6.0]]
}

transitions {
  infection[a in age] : S[a] --> E[a]
    @ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
  progression[a in age] : E[a] --> I[a]  @ sigma * E[a]
  recovery[a in age]    : I[a] --> R[a]  @ gamma * I[a]
}

events {
  # School entry: on a fixed day each year a fraction of every young
  # compartment ages into the old one — four transfers at one instant, each
  # reading the pre-event state. Mid-step instants at dt = 1.
  age_S : transfer(fraction = frac_age, from = S[young], to = S[old]) at [60.5, 120.5]
  age_E : transfer(fraction = frac_age, from = E[young], to = E[old]) at [60.5, 120.5]
  age_I : transfer(fraction = frac_age, from = I[young], to = I[old]) at [60.5, 120.5]
  age_R : transfer(fraction = frac_age, from = R[young], to = R[old]) at [60.5, 120.5]
  # A number of infectious adults imported, mid-step.
  import : add(I[old], seed) at [30.5, 90.5]
}

observations {
  cases {
    columns   { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    emit_schedule = every 7 'days
    projected     = sum(a in age, incidence(infection[a]))
    cases ~ poisson(rate = projected)
  }
}

init {
  S[young] = 3998
  S[old]   = 6000
  I[young] = 2
}

simulate { from = 0 'days  to = 154 'days }
"#;

/// Compile, simulate weekly cases at the truth, return `(tmp, ir, data)`.
fn setup(tag: &str, model_src: &str) -> Option<(TempDir, PathBuf, PathBuf)> {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return None;
    }
    let tmp = tempdir(tag);
    let model_path = tmp.path().join("model.camdl");
    std::fs::write(&model_path, model_src).unwrap();
    let ir = tmp.path().join("model.ir.json");
    let out = Command::new(camdlc().unwrap()).env_remove("CLICOLOR_FORCE").arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();

    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, "beta = 0.02\nsigma = 0.2\ngamma = 0.1\nseed = 40\nfrac_age = 0.2\n").unwrap();
    let data = tmp.path().join("cases.tsv");
    let sim = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "7", "--obs-only"]).arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));
    Some((tmp, ir, data))
}

fn fit_toml(out_dir: &Path, ir: &Path, data: &Path, algorithm: &str) -> String {
    format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[estimate]
seed = {{ bounds = [0.5, 200.0], start = 20.0 }}
frac_age = {{ bounds = [0.02, 0.8], start = 0.4 }}
[fixed]
beta = 0.02
sigma = 0.2
gamma = 0.1
[method]
algorithm = "{algorithm}"
backend = "ode"
chains = 1
starts = "single"
tolerance = 1e-9
max_evals = 5000
"#, out = out_dir.display(), ir = ir.display(), data = data.display())
}

fn find_file(dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().and_then(|n| n.to_str()) == Some(name) { out.push(p); }
            }
        }
    }
    out
}

/// `(loglik, n_evals, seed, frac_age)` of the single chain in `chain_results.tsv`.
fn chain_result(out_dir: &Path) -> (f64, usize, f64, f64) {
    let files = find_file(out_dir, "chain_results.tsv");
    assert_eq!(files.len(), 1, "expected one chain_results.tsv under {}", out_dir.display());
    let text = std::fs::read_to_string(&files[0]).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let row: Vec<&str> = lines.next().expect("one chain row").split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap_or_else(|| panic!("column {name}"));
    let num = |name: &str| row[col(name)].parse::<f64>().unwrap_or_else(|e| panic!("{name}: {e}"));
    let status = row[col("status")];
    assert!(
        status != "failed" && !status.starts_with("max"),
        "the chain must have converged (an NLopt success code), got status {status}"
    );
    (num("loglik"), num("n_evals") as usize, num("seed"), num("frac_age"))
}

fn run_fit(bin: &Path, fit: &Path) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(["fit", "run"]).arg(fit)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap()
}

#[test]
fn nl_lbfgs_through_events_reaches_the_sbplx_optimum_in_fewer_evaluations() {
    let Some((tmp, ir, data)) = setup("fit", MODEL) else { return };
    let bin = camdl_bin();

    let mut results = Vec::new();
    for algorithm in ["nl-lbfgs", "nl-sbplx"] {
        let out_dir = tmp.path().join(format!("out_{algorithm}"));
        let fit = tmp.path().join(format!("fit_{algorithm}.toml"));
        std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, algorithm)).unwrap();
        let out = run_fit(&bin, &fit);
        let msg = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "{algorithm} fit must succeed (exit 0):\n{msg}");
        assert!(
            !msg.contains("does not yet support scheduled interventions or events"),
            "{algorithm}: the event-jump refusal must be gone:\n{msg}"
        );
        results.push((algorithm, chain_result(&out_dir)));
    }
    let (_, (ll_l, n_l, seed_l, frac_l)) = results[0];
    let (_, (ll_s, n_s, seed_s, frac_s)) = results[1];
    eprintln!(
        "nl-lbfgs: ll = {ll_l:.6}, seed = {seed_l:.4}, frac_age = {frac_l:.6}, {n_l} evaluations\n\
         nl-sbplx: ll = {ll_s:.6}, seed = {seed_s:.4}, frac_age = {frac_s:.6}, {n_s} evaluations"
    );
    // Same optimum: the log-likelihood to 0.01 and each coordinate to 0.5 % of
    // its value (the optimizers stop by `xtol_rel`, so they agree in θ, not to
    // the last bit).
    assert!((ll_l - ll_s).abs() < 0.01, "optima differ in loglik: {ll_l} vs {ll_s}");
    assert!((seed_l - seed_s).abs() < 5e-3 * seed_s, "seed differs: {seed_l} vs {seed_s}");
    assert!((frac_l - frac_s).abs() < 5e-3 * frac_s, "frac_age differs: {frac_l} vs {frac_s}");
    // The gradient method needs fewer objective evaluations (one evaluation is
    // one augmented solve for `nl-lbfgs`, one plain solve for `nl-sbplx`).
    assert!(n_l < n_s, "nl-lbfgs took {n_l} evaluations, nl-sbplx {n_s}");
}

#[test]
fn a_fixed_count_transfer_is_refused_at_config_validation_not_per_chain() {
    let counted = MODEL.replace(
        "import : add(I[old], seed) at [30.5, 90.5]",
        "import : transfer(count = seed, from = S[old], to = I[old]) at [30.5, 90.5]",
    );
    let Some((tmp, ir, data)) = setup("refuse", &counted) else { return };
    let bin = camdl_bin();
    let out_dir = tmp.path().join("out");
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, "nl-lbfgs")).unwrap();
    let out = run_fit(&bin, &fit);
    let msg = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
    assert!(!out.status.success(), "a fixed-count transfer under nl-lbfgs must be refused:\n{msg}");
    assert!(
        msg.contains("import") && msg.contains("transfer(count") && msg.contains("nl-sbplx"),
        "the refusal must name the event, the construct and the alternative:\n{msg}"
    );
    assert!(
        !msg.contains("no finite log-likelihood") && !msg.contains("impossible region"),
        "a refusal must not be reported as a starting-value problem:\n{msg}"
    );
    assert!(
        find_file(&out_dir, "chain_results.tsv").is_empty(),
        "no method leaf may be written for a refused fit"
    );

    // Control: the derivative-free optimizer on the same model has no such
    // requirement and runs.
    let out_dir = tmp.path().join("out_sbplx");
    let fit = tmp.path().join("fit_sbplx.toml");
    std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, "nl-sbplx")).unwrap();
    let out = run_fit(&bin, &fit);
    assert!(out.status.success(), "nl-sbplx has no gradient requirement:\n{}", String::from_utf8_lossy(&out.stderr));
}
