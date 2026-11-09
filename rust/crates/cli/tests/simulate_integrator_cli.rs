//! gh#166 (PR #231 review): end-to-end `camdl simulate --integrator` override.
//!
//! Asserts the CLI flag actually reaches the integrator: `--integrator rk45`
//! produces a DIFFERENT trajectory than `--integrator rk4` on the same model,
//! the default (no flag) matches `rk4`, and `--integrator rk45` PRESERVES a
//! model-declared tolerance (method-only override — documented precedence).
//!
//! Shells out to the built `camdl` binary; skipped silently when the release
//! binary or `camdlc.exe` isn't present (rust-only CI / pre-build).

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
    let base = std::env::temp_dir().join(format!("camdl_integcli_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// Author + compile a SIR whose `simulate {}` block is `simulate_clause`
/// (e.g. with or without a declared `integrator`). Returns the compiled IR path.
fn compile_sir(dir: &Path, camdlc: &Path, simulate_clause: &str) -> PathBuf {
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
init {{ S = 9990  I = 10 }}
{simulate_clause}
"#);
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, &src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(camdlc).env_remove("CLICOLOR_FORCE").arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Run `camdl simulate <ir> --backend ode --dt 1 [--integrator <m>] --stdout`,
/// returning the trajectory TSV captured from stdout.
fn simulate_stdout(bin: &Path, ir: &Path, params: &Path, integrator: Option<&str>) -> String {
    let mut cmd = Command::new(bin);
    cmd.env_remove("CLICOLOR_FORCE");
    cmd.args(["simulate"]).arg(ir)
        .args(["--params"]).arg(params)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--stdout"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1");
    if let Some(m) = integrator {
        cmd.args(["--integrator", m]);
    }
    let out = cmd.output().unwrap();
    assert!(out.status.success(), "simulate failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn simulate_integrator_override_changes_trajectory() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("override");
    // Model declares NO integrator → defaults to rk4.
    let ir = compile_sir(tmp.path(), &camdlc().unwrap(), "simulate { from = 0 'days  to = 60 'days }");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.5\ngamma = 0.25\nN0 = 10000\n").unwrap();

    let rk4 = simulate_stdout(&bin, &ir, &params, Some("rk4"));
    let rk45 = simulate_stdout(&bin, &ir, &params, Some("rk45"));
    let default = simulate_stdout(&bin, &ir, &params, None);

    assert!(!rk4.is_empty() && !rk45.is_empty(), "trajectories must be non-empty");
    // The override reaches the integrator: rk4 (fixed dt=1) and rk45 (adaptive)
    // produce measurably different trajectories.
    assert_ne!(rk4, rk45, "`--integrator rk45` must change the trajectory vs `rk4`");
    // No flag == the model default (rk4).
    assert_eq!(default, rk4, "default (no --integrator) must match the model's rk4 default");
}

#[test]
fn simulate_integrator_rk45_preserves_declared_tolerance() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("preserve");
    // Model DECLARES a tight rk45 tolerance.
    let ir = compile_sir(
        tmp.path(),
        &camdlc().unwrap(),
        "simulate { from = 0 'days  to = 60 'days  integrator = rk45 { atol = 1e-11  rtol = 1e-11 } }",
    );
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.5\ngamma = 0.25\nN0 = 10000\n").unwrap();

    // `--integrator rk45` is method-only: it must PRESERVE the model's declared
    // tolerance, so the output equals the no-override run.
    let declared = simulate_stdout(&bin, &ir, &params, None);
    let overridden = simulate_stdout(&bin, &ir, &params, Some("rk45"));
    assert_eq!(
        declared, overridden,
        "`--integrator rk45` must preserve the model-declared tolerance (method-only override)"
    );
    // Overriding to rk4 drops the adaptive integrator → different trajectory.
    let to_rk4 = simulate_stdout(&bin, &ir, &params, Some("rk4"));
    assert_ne!(declared, to_rk4, "`--integrator rk4` must override the declared rk45");
}
