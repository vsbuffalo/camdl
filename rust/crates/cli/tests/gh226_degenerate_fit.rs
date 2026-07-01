//! gh#226 — a degenerate (all-`-inf`) fit must fail loudly, not complete with
//! exit 0 and a silently degenerate posterior.
//!
//! DRIVER-level end-to-end coverage for the whole-fit backstop wired into the
//! three fit drivers (`pmmh::run_stage`, IF2's
//! `runner::run_chains_with_per_chain_params`, `pgas::run_stage`). The
//! sim-level tests (`sim/tests/gh226_inf_loglik_backstop.rs`) pin the shared
//! `no_finite_anchor` predicate and the raw `run_pmmh` result, but they do NOT
//! prove the DRIVER now returns `Err` and the *process* exits non-zero. This
//! test drives `camdl fit run` end-to-end and asserts exactly that.
//!
//! ## The impossible-data mechanism
//!
//! `binom_logpmf` (obs_loglik.rs) returns `NEG_INFINITY` whenever the observed
//! `k > n`. The model observes `cases ~ binomial(n = prevalence(I), p = 0.5)`,
//! so `n` is bounded by the reachable I compartment (≤ a few hundred for this
//! tiny SIR). The data sets `cases = 1_000_000` at every observation time —
//! far above any reachable count — so *every* particle at *every* θ scores
//! `-inf`. The likelihood surface is uniformly `-inf`, independent of
//! parameters and seed: a guaranteed degenerate fit.
//!
//! The series is deliberately **2 observations long**. An all-`-inf` window has
//! ESS 0, and the PF's degeneracy watchdog (`check_pf_degeneracy`) trips
//! `EssCollapsed` only after `ESS_COLLAPSE_WINDOWS = 3` *consecutive* collapsed
//! windows. With two windows the watchdog never fires (and `dead_count` stays 0
//! — no per-particle step error — so `AllParticlesDead` never fires either), so
//! the filter returns `Ok(-inf)` rather than `Err(PFDegenerate)`. That is
//! precisely the gh#226 path: a *finite-structure* result whose loglik is
//! `-inf`, which the old drivers accepted (writing a degenerate fit_state and
//! exiting 0) and which the backstop must now reject.
//!
//! ## Acceptance (per method: pmmh / if2 / pgas)
//!   1. `camdl fit run` exits NON-ZERO.
//!   2. a `diagnostics.json` under the output tree carries an
//!      `initial_loglik_infinite` entry (the snake_case serde tag of
//!      `DiagnosticKind::InitialLoglikInfinite`).
//!
//! Silent-skip when the release binary / camdlc is not built (mirrors
//! `pmmh_bad_init_skip` / `fit_sparse_holes`).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../target/release/camdl");
    if p.exists() { Some(p) } else { None }
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
        "camdl_gh226_degen_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Tiny SIR whose `cases` stream is `binomial(n = prevalence(I), p = 0.5)`.
/// `n` is bounded by the population, so an observed `cases` far above any
/// reachable count scores `-inf` at every θ (obs_loglik `k > n` branch).
fn write_model(dir: &Path, camdlc: &Path) -> PathBuf {
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
    cases ~ binomial(n = projected, p = 0.5)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 2 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Two impossible observations (`cases = 1_000_000` ≫ any reachable I). Two
/// rows keeps the PF degeneracy watchdog (K=3 consecutive collapsed windows)
/// from firing, so the surface stays `Ok(-inf)`, not `Err(PFDegenerate)`.
fn write_impossible_data(dir: &Path) -> PathBuf {
    let p = dir.join("cases.tsv");
    std::fs::write(&p, "time\tcases\n1\t1000000\n2\t1000000\n").unwrap();
    p
}

/// The per-method stage block. Every method starts from the same fixed `start`
/// values (`init = "single"`), tiny particle/iteration counts (the assertion is
/// "backstop fires", not convergence).
fn stage_block(method: &str) -> String {
    match method {
        "pmmh" => r#"[stages.post]
algorithm = "pmmh"
backend   = "chain_binomial"
chains    = 1
particles = 20
iterations = 10
burn_in   = 2
thin      = 1
init      = "single"
"#.to_string(),
        "if2" => r#"[stages.post]
algorithm = "if2"
backend   = "chain_binomial"
chains    = 1
particles = 20
iterations = 1
cooling   = 0.9
init      = "single"
"#.to_string(),
        "pgas" => r#"[stages.post]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 1
particles = 20
sweeps    = 5
burn_in   = 1
thin      = 1
init      = "single"
"#.to_string(),
        other => panic!("unknown method {other}"),
    }
}

fn write_fit_toml(dir: &Path, ir: &Path, data: &Path, method: &str) -> (PathBuf, PathBuf) {
    let out_root = dir.join("results");
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.3 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.1 }}
[fixed]
N0 = 1000
{stage}"#,
        out = out_root.display(),
        ir = ir.display(),
        data = data.display(),
        stage = stage_block(method));
    let p = dir.join("fit.toml");
    std::fs::write(&p, toml).unwrap();
    (p, out_root)
}

/// Recursively collect every file named `diagnostics.json` under `root`.
/// The failing stage errors out BEFORE CAS finalize, so its diagnostics land
/// in the streaming-claim leaf under the output tree — not at a predictable
/// committed leaf. Walking the whole tree finds it wherever it is.
fn find_diagnostics(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|n| n == "diagnostics.json") {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// True when any `diagnostics.json` under `root` carries an
/// `initial_loglik_infinite` entry (the snake_case serde tag of
/// `DiagnosticKind::InitialLoglikInfinite`).
fn has_initial_loglik_infinite(root: &Path) -> bool {
    for path in find_diagnostics(root) {
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else { continue };
        let Some(arr) = v.as_array() else { continue };
        if arr.iter().any(|d| {
            d.get("kind")
                .and_then(|k| k.get("type"))
                .and_then(|t| t.as_str())
                == Some("initial_loglik_infinite")
        }) {
            return true;
        }
    }
    false
}

/// Drive `camdl fit run` on the impossible-data model for `method` and assert
/// the driver backstop fires: non-zero exit + an `initial_loglik_infinite`
/// diagnostic. Skips when the release binary / camdlc is not built.
fn assert_degenerate_fit_is_rejected(method: &str) {
    let (Some(bin), Some(camdlc)) = (camdl_bin(), camdlc_bin()) else {
        eprintln!("skip: release camdl / camdlc not built");
        return;
    };
    let tmp = tempdir(method);
    let ir = write_model(tmp.path(), &camdlc);
    let data = write_impossible_data(tmp.path());
    let (toml, out_root) = write_fit_toml(tmp.path(), &ir, &data, method);

    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &toml.to_string_lossy(),
               "--seed", "1", "--progress", "none"])
        .output()
        .expect("spawn camdl fit run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Acceptance 1: NON-ZERO exit. The old drivers accepted the degenerate
    // `-inf` result and exited 0.
    assert!(!out.status.success(),
        "[{method}] a degenerate all-(-inf) fit must exit NON-ZERO (gh#226); \
         got success.\nstdout:\n{stdout}\nstderr:\n{stderr}");

    // Acceptance 2: an `initial_loglik_infinite` diagnostic was written.
    assert!(has_initial_loglik_infinite(&out_root),
        "[{method}] a degenerate fit must write an `initial_loglik_infinite` \
         diagnostic under {}.\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out_root.display());
}

#[test]
fn pmmh_degenerate_fit_exits_nonzero_with_diagnostic() {
    assert_degenerate_fit_is_rejected("pmmh");
}

#[test]
fn if2_degenerate_fit_exits_nonzero_with_diagnostic() {
    assert_degenerate_fit_is_rejected("if2");
}

#[test]
fn pgas_degenerate_fit_exits_nonzero_with_diagnostic() {
    assert_degenerate_fit_is_rejected("pgas");
}
