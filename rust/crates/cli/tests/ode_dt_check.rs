//! Wiring guard for the deterministic ODE Richardson dt-convergence check
//! (gh#52, gh#227) — `dt_check::run_richardson_ladder_ode`, wired into the
//! `nl-sbplx`/`nl-bobyqa` (ODE-MLE) and `mh` (ODE-MH) stages.
//!
//! The bug this guards: before gh#227 the ODE inference stages wrote
//! `dt_check: None` — the post-fit dt-convergence audit silently never ran, so
//! a discretization-dependent MLE/MAP (coarse dt creating a fake basin that
//! synthetic recovery shares and can't detect) shipped with no warning. This
//! test asserts the audit now RUNS on the ODE path and writes a verdict to
//! `fit_state.toml`, and that `--no-dt-check` suppresses it.
//!
//! End-to-end via the built `camdl` binary; skipped silently when the release
//! binary or `camdlc.exe` isn't present so the suite stays runnable in
//! rust-only CI and before a build.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_odedtcheck_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// SIR on the ODE backend with a prevalence observation; beta estimated,
/// gamma + N0 fixed. Returns the compiled IR path.
fn write_model(dir: &Path, camdl: &Path) -> PathBuf {
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0] ~ log_normal(mu = 0.0, sigma = 1.0)
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(camdl).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Synthetic prevalence observations from the ODE backend at a known beta.
fn gen_data(bin: &Path, ir: &Path, dir: &Path) -> PathBuf {
    let truth = dir.join("truth.toml");
    std::fs::write(&truth, "beta = 0.8\ngamma = 0.3\nN0 = 10000\n").unwrap();
    let data = dir.join("cases.tsv");
    let sim = Command::new(bin)
        .args(["simulate"]).arg(ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"]).arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(),
        "simulate (data gen) failed: {}", String::from_utf8_lossy(&sim.stderr));
    assert!(data.exists(), "synthetic data not written");
    data
}

/// Walk `dir` for every `fit_state.toml` and return their contents.
fn find_fit_states(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().and_then(|n| n.to_str()) == Some("fit_state.toml") {
                    if let Ok(text) = std::fs::read_to_string(&p) { found.push(text); }
                }
            }
        }
    }
    found
}

/// Run an `nl-sbplx` ODE fit. `extra_args` is appended to `fit run` (e.g.
/// `--no-dt-check`). Returns the concatenated fit_state.toml contents.
fn run_nlsbplx_fit(tag: &str, extra_args: &[&str]) -> Option<String> {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return None;
    }
    let tmp = tempdir(tag);
    let ir = write_model(tmp.path(), &camdlc().unwrap());
    let data = gen_data(&bin, &ir, tmp.path());

    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = 1.5 }}

[fixed]
gamma = 0.3
N0 = 10000

[stages.mle]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#, out = out.display(), ir = ir.display(), data = data.display())).unwrap();

    let mut cmd = Command::new(&bin);
    cmd.args(["fit", "run"]).arg(&fit_toml);
    for a in extra_args { cmd.arg(a); }
    let status = cmd.env("CAMDL_SKIP_VERSION_CHECK", "1").status().unwrap();
    assert!(status.success(), "nl-sbplx ODE `fit run {tag}` must succeed (exit 0)");

    let states = find_fit_states(&out);
    assert!(!states.is_empty(), "no fit_state.toml produced under {}", out.display());
    Some(states.join("\n---\n"))
}

#[test]
fn ode_dt_check_runs_and_writes_verdict() {
    let Some(text) = run_nlsbplx_fit("on", &[]) else { return };
    // The dt-check ran: fit_state.toml carries a [dt_check] block with a
    // non-skipped verdict. (Before gh#227 this block was absent —
    // `dt_check: None` on every ODE stage.)
    assert!(text.contains("[dt_check]"),
        "fit_state.toml has no [dt_check] block — the ODE dt-check did not run \
         (regression: gh#227 wiring). Contents:\n{text}");
    let verdict_ok = ["verdict = \"pass\"", "verdict = \"marginal\"", "verdict = \"fail\""]
        .iter().any(|v| text.contains(v));
    assert!(verdict_ok,
        "[dt_check] present but no pass/marginal/fail verdict found:\n{text}");
}

#[test]
fn ode_dt_check_suppressed_by_no_dt_check_flag() {
    // --no-dt-check requires --stage since it is keyed into that stage's
    // identity via CliStageOverrides (gh#540 seam; gh#726 for the nl-*
    // dt_check field).
    let Some(text) = run_nlsbplx_fit("off", &["--no-dt-check", "--stage", "mle"]) else { return };
    // --no-dt-check → enabled=false → Skipped → the block is omitted entirely
    // (mirrors the IF2 path's "skipped omits the block" semantics).
    assert!(!text.contains("[dt_check]"),
        "--no-dt-check should suppress the [dt_check] block, but it is present:\n{text}");
}
