//! Run-identity parity between simulate's run path and its identity path
//! (audit 2026-08-23 #1/#2; proposal
//! docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md Phase 1).
//!
//! `resolve_run_model` applies `--integrator` and `--param-vec` to the model
//! that is RUN; `build_simulate_cas_sink` builds the model that is HASHED in
//! a second load. Before the fix the second load skipped both overrides, so
//! two runs with different settings shared one `run_id` — the second was
//! silently served the first's trajectory (pre-S1), or died with
//! DivergentRecompute (post-S1). These tests pin that each override splits
//! the leaves: two settings, two CAS leaves, both runs succeeding.
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
    let base = std::env::temp_dir()
        .join(format!("camdl_idparity_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

fn compile(dir: &Path, camdlc: &Path, src: &str, stem: &str) -> PathBuf {
    let model_path = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join(format!("{stem}.ir.json"));
    let out = Command::new(camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Count committed sim leaves (dirs holding a run.json) under `<out>/sims`.
fn sim_leaves(out: &Path) -> Vec<PathBuf> {
    let mut leaves = Vec::new();
    let mut stack = vec![out.join("sims")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").exists() {
                    leaves.push(p);
                } else {
                    stack.push(p);
                }
            }
        }
    }
    leaves
}

const SIR_ODE: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;

#[test]
fn integrator_override_splits_the_cas_leaves() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("integrator");
    let ir = compile(tmp.path(), &cc, SIR_ODE, "sir");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.5\ngamma = 0.25\nN0 = 10000\n").unwrap();
    let out = tmp.path().join("out");

    for m in ["rk4", "rk45"] {
        let st = Command::new(&bin)
            .args(["simulate"]).arg(&ir)
            .args(["--params"]).arg(&params)
            .args(["--backend", "ode", "--dt", "1", "--seed", "1",
                   "--integrator", m,
                   "--cas", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("traj_{m}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().unwrap();
        assert!(st.status.success(),
            "simulate --integrator {m} must succeed (a DivergentRecompute here \
             means the override is not in the run identity): {}",
            String::from_utf8_lossy(&st.stderr));
    }
    assert_eq!(sim_leaves(&out).len(), 2,
        "rk4 and rk45 runs must land in two distinct CAS leaves");
}

const SIR_VEC: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta_a : rate  in [0.05, 5.0]
  beta_b : rate  in [0.05, 5.0]
  gamma  : rate  in [0.01, 1.0]
  N0     : count in [100, 100000]
}
transitions {
  infection_a : S --> I @ beta_a * S * I / N0
  infection_b : S --> I @ beta_b * S * I / N0
  recovery    : I --> R @ gamma * I
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 30 'days }
"#;

#[test]
fn param_vec_values_split_the_cas_leaves() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("paramvec");
    let ir = compile(tmp.path(), &cc, SIR_VEC, "sirvec");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "gamma = 0.25\nN0 = 10000\n").unwrap();
    let out = tmp.path().join("out");

    for (tag, a, b) in [("v1", 0.4, 0.6), ("v2", 0.9, 1.1)] {
        let vec_file = tmp.path().join(format!("beta_{tag}.tsv"));
        std::fs::write(&vec_file, format!("a\t{a}\nb\t{b}\n")).unwrap();
        let st = Command::new(&bin)
            .args(["simulate"]).arg(&ir)
            .args(["--params"]).arg(&params)
            .args(["--seed", "1",
                   "--param-vec"]).arg(format!("beta={}", vec_file.display()))
            .args(["--cas", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("traj_{tag}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().unwrap();
        assert!(st.status.success(),
            "simulate --param-vec ({tag}) must succeed (a DivergentRecompute \
             here means the vec values are not in the run identity): {}",
            String::from_utf8_lossy(&st.stderr));
    }
    assert_eq!(sim_leaves(&out).len(), 2,
        "two different --param-vec files must land in two distinct CAS leaves");
}
