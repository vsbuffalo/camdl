//! PGAS runs its NUTS θ-move only where the model has rate gradients; on a
//! model where no parameter reaches a transition rate it falls back to
//! Metropolis-within-Gibbs. An explicit `use_nuts = true` is then refused;
//! the default falls back with a notice.
//!
//! End-to-end via the built `camdl` binary. A missing release binary or
//! `camdlc.exe` fails the test (run `make build`, or `make test-fast`, which
//! builds both) rather than passing it without running.

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

/// Fit a model whose only estimated parameter (`rho`) is an observation
/// rate: the transition rates use typed constants, so the IR carries no
/// rate gradient. Returns (exit success, stderr).
fn run(tag: &str, use_nuts_line: &str) -> (bool, String) {
    let camdlc = camdlc().expect(
        "camdlc.exe is missing: run `make build-ocaml` (this test must not pass without running)");
    assert!(camdl_bin().exists(),
        "the release camdl binary is missing: run `make build-rust` (this test must not pass \
         without running)");
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("camdl_nuts_nograd_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("m.camdl");
    std::fs::write(&model, r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  rho : probability in [0.01, 1.0]
}
let beta : rate = 0.3
let gamma : rate = 0.1
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    cases ~ poisson(rate = rho * projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 20 'days }
"#).unwrap();
    let out = Command::new(&camdlc).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    let mut tsv = String::from("time\tcases\n");
    for (k, c) in [6, 8, 11, 14, 19, 25, 32, 40, 49, 58].iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", 2 * (k + 1), c));
    }
    std::fs::write(dir.join("cases.tsv"), tsv).unwrap();
    let fit = dir.join("fit.toml");
    std::fs::write(&fit, format!(r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
rho = {{ prior = {{ beta = {{ alpha = 2.0, beta = 2.0 }} }} }}

[fixed]

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 20
sweeps = 6
burn_in = 2
{use_nuts_line}
"#, out = dir.join("out").display(), ir = ir.display(),
        data = dir.join("cases.tsv").display())).unwrap();
    let out = Command::new(camdl_bin())
        .args(["fit", "run"]).arg(&fit).arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn an_explicit_use_nuts_without_gradients_is_refused() {
    let (ok, err) = run("explicit", "use_nuts = true");
    assert!(!ok, "use_nuts = true on a gradient-free model must be refused:\n{err}");
    assert!(err.contains("use_nuts = true") && err.contains("rate gradients"), "{err}");
}

#[test]
fn the_default_falls_back_with_a_notice() {
    let (ok, err) = run("default", "");
    assert!(ok, "the default runs, by Metropolis-within-Gibbs:\n{err}");
    assert!(err.contains("Metropolis-within-Gibbs, not NUTS"), "says so:\n{err}");
}

#[test]
fn nuts_tuning_on_the_fallback_is_refused() {
    let (ok, err) = run("tuning", "max_tree_depth = 8");
    assert!(!ok, "max_tree_depth on the Metropolis-within-Gibbs fallback must be refused:\n{err}");
    assert!(err.contains("max_tree_depth") && err.contains("Metropolis-within-Gibbs"), "{err}");
}
