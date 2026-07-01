//! Safety guard for `algorithm = "mh"` on `backend = "ode"` — deterministic
//! Metropolis-Hastings over the ODE marginal likelihood (Phase 1 of
//! docs/dev/proposals/2026-06-15-ode-gradient-inference.md).
//!
//! End-to-end: author a small SIR, simulate synthetic prevalence observations
//! at a KNOWN beta on the ODE backend, then run an `mh`+`ode` fit starting far
//! from the truth and assert the posterior recovers it. This guards the whole
//! mh-on-ode path — the `Stage::Mh` config variant, the dispatch arm, and the
//! deterministic `compute_ode_loglik` eval seam in `pmmh::run_stage` — against
//! "wiring broken / degenerate / silently-wrong" regressions. "Works as
//! expected" = the chain moves from the start toward the truth and produces a
//! finite, in-bounds posterior.
//!
//! Shells out to the built `camdl` binary; skipped silently when the release
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
        "camdl_mhode_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

const TRUE_BETA: f64 = 0.8;
// Far BELOW truth (0.8), not above. A start above truth (e.g. 1.8, R0=6) burns
// the epidemic out — I(t) reaches exactly 0 well before the data ends, so
// prevalence-observed data with ongoing late cases scores -inf at every θ. That
// is a genuinely degenerate fit: the chain starts at -inf and can never move
// (log_alpha = finite - (-inf) = +inf, rejected), which gh#226's backstop now
// correctly rejects. 0.4 (R0=1.33) is a sustained-but-weak epidemic that keeps
// prevalence > 0, so the init loglik is finite and the chain genuinely climbs
// toward 0.8 — a real recovery test, still far from truth.
const START_BETA: f64 = 0.4;

/// SIR with a weakly-informative `~` prior on beta; gamma + N0 fixed. R0 ≈ 2.7
/// over 60 days gives a clear epidemic, so the observations identify beta well.
/// `projected` selects the observation projection (`prevalence(I)` or
/// `incidence(infection)`), which is the only thing the two recovery tests
/// differ in. Returns the compiled IR path.
fn write_model_proj(dir: &Path, camdl: &Path, projected: &str, integ: &str) -> PathBuf {
    let src = format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate  in [0.05, 5.0] ~ log_normal(mu = 0.0, sigma = 1.0)
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}}
transitions {{
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}}
observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = {projected}
    emit_schedule = every 2 'days
    cases ~ poisson(rate = projected)
  }}
}}
init {{ S = 9990  I = 10 }}
simulate {{ from = 0 'days  to = 60 'days  {integ} }}
"#);
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, &src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(camdl).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Walk `dir` for every file named `name`.
fn find_named(dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
                    found.push(p);
                }
            }
        }
    }
    found
}

/// Walk `dir` for every `trace.tsv`. The mh stage writes one per chain at
/// `<stage_dir>/chain_<N>/trace.tsv`.
fn find_traces(dir: &Path) -> Vec<PathBuf> {
    find_named(dir, "trace.tsv")
}

/// Post-burn-in beta samples from a trace.tsv. Columns are
/// `step, log_likelihood, log_posterior, accepted, <params...>`; the beta
/// column is resolved by header name, not position.
fn beta_samples(trace: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(trace).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header = lines.next().expect("trace header");
    let beta_col = header.split('\t').position(|h| h == "beta")
        .unwrap_or_else(|| panic!("no `beta` column in trace header: {header}"));
    lines.filter_map(|l| l.split('\t').nth(beta_col).and_then(|v| v.parse::<f64>().ok()))
        .collect()
}

/// Shared end-to-end recovery harness: author the SIR with the given observation
/// `projected`, simulate synthetic data at TRUE_BETA on the ODE backend, run an
/// `mh`+`ode` fit from START_BETA, and return (posterior beta mean, out dir,
/// tmp). The tmp guard must be kept alive by the caller. Returns `None` when the
/// release binary / camdlc is missing (so the test skips).
fn recover_beta(tag: &str, projected: &str) -> Option<(f64, PathBuf, TempDir)> {
    recover_beta_integ(tag, projected, "")
}

/// As `recover_beta`, but with an explicit `simulate {}` integrator clause
/// (e.g. `integrator = rk45 { atol = 1e-8  rtol = 1e-6 }`). The clause is honored
/// for BOTH the synthetic data-gen and the fit (`compute_ode_loglik` reads the
/// model's declared integrator), so this drives rk45 through the whole fit path.
fn recover_beta_integ(tag: &str, projected: &str, integ: &str) -> Option<(f64, PathBuf, TempDir)> {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return None;
    }
    let tmp = tempdir(tag);
    let ir = write_model_proj(tmp.path(), &camdlc().unwrap(), projected, integ);

    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, format!("beta = {TRUE_BETA}\ngamma = 0.3\nN0 = 10000\n")).unwrap();
    let data = tmp.path().join("cases.tsv");
    let sim = Command::new(&bin)
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"]).arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(),
        "simulate (data gen) failed: {}", String::from_utf8_lossy(&sim.stderr));
    assert!(data.exists(), "synthetic data not written");

    let out = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = {START_BETA} }}

[fixed]
gamma = 0.3
N0 = 10000

[stages.mh]
algorithm = "mh"
backend = "ode"
init = "single"
chains = 2
iterations = 1500
burn_in = 400
"#, out = out.display(), ir = ir.display(), data = data.display())).unwrap();

    let status = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit_toml)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status().unwrap();
    assert!(status.success(), "mh+ode `fit run` must succeed (exit 0)");

    let traces = find_traces(&out);
    assert!(!traces.is_empty(), "no chain trace.tsv produced under {}", out.display());
    let betas: Vec<f64> = traces.iter().flat_map(|t| beta_samples(t)).collect();
    assert!(betas.len() >= 100,
        "too few post-burn-in beta samples ({}) — the mh chains didn't run", betas.len());
    let mean = betas.iter().sum::<f64>() / betas.len() as f64;
    Some((mean, out, tmp))
}

#[test]
fn mh_ode_recovers_known_beta() {
    let Some((mean, out, _tmp)) = recover_beta("recovery", "prevalence(I)") else { return };

    assert!(mean.is_finite() && (0.05..=5.0).contains(&mean),
        "posterior beta mean {mean} not finite/in-bounds");
    // Moved from the start (1.8) toward the truth (0.8): a generous band that
    // catches a stuck/degenerate/silently-wrong chain without being flaky.
    assert!((0.45..=1.35).contains(&mean),
        "mh+ode did not recover beta: posterior mean {mean:.3} (truth {TRUE_BETA}, \
         start {START_BETA}). Expected the chain to move from {START_BETA} toward \
         {TRUE_BETA}; a mean near {START_BETA} means it never moved, near a bound \
         means it degenerated.");

    // gh#52, gh#227: the deterministic ODE dt-check ran at the MAP and wrote a
    // verdict to fit_state.toml. Before gh#227 the mh path wrote `dt_check:
    // None` — the audit silently never ran on ODE-MH. Guard that it does.
    let states = find_named(&out, "fit_state.toml");
    assert!(!states.is_empty(), "no fit_state.toml produced under {}", out.display());
    let any_dt_check = states.iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .any(|t| t.contains("[dt_check]"));
    assert!(any_dt_check,
        "mh+ode fit_state.toml has no [dt_check] block — the ODE dt-check did \
         not run on the mh path (regression: gh#227 wiring).");
}

/// gh#166 Phase B (B7): the same recovery, but the observation projects
/// `incidence(infection)` — so the fit scores through the AUGMENTED per-interval
/// flow that `compute_ode_loglik` sums from `snapshot.flows`. Recovering beta
/// from incidence data end-to-end proves the augmented (high-order) flow is what
/// the ODE-inference likelihood now consumes, and that the incidence-scoring
/// shape (sum per-interval flows, fold, score) still works.
#[test]
fn mh_ode_recovers_known_beta_from_incidence() {
    let Some((mean, _out, _tmp)) = recover_beta("incidence", "incidence(infection)") else { return };

    assert!(mean.is_finite() && (0.05..=5.0).contains(&mean),
        "incidence posterior beta mean {mean} not finite/in-bounds");
    assert!((0.45..=1.35).contains(&mean),
        "mh+ode did not recover beta from INCIDENCE observations: posterior mean \
         {mean:.3} (truth {TRUE_BETA}, start {START_BETA}). A failure here means the \
         augmented incidence is not flowing correctly through compute_ode_loglik.");
}

/// gh#166 (PR #231 review): no test ran rk45 through `camdl fit`. Same recovery,
/// but the model DECLARES `integrator = rk45 { ... }`, so both the synthetic
/// data-gen and the `compute_ode_loglik` fit seam run the adaptive integrator.
/// Recovering beta end-to-end proves rk45 is honored on the inference path (it
/// reads `model.simulation.integrator`), not just in forward `simulate`.
#[test]
fn mh_ode_recovers_known_beta_rk45() {
    let Some((mean, _out, _tmp)) = recover_beta_integ(
        "recovery_rk45",
        "prevalence(I)",
        "integrator = rk45 { atol = 1e-8  rtol = 1e-6 }",
    ) else { return };

    assert!(mean.is_finite() && (0.05..=5.0).contains(&mean),
        "rk45 mh+ode posterior beta mean {mean} not finite/in-bounds");
    assert!((0.45..=1.35).contains(&mean),
        "mh+ode on a model declaring `integrator = rk45` did not recover beta: \
         posterior mean {mean:.3} (truth {TRUE_BETA}, start {START_BETA}). A failure \
         here means rk45 is not honored on the fit path (compute_ode_loglik).");
}
