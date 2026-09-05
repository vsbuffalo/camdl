//! End-to-end: `camdl fit --method nuts --backend ode` recovers a known `beta`
//! from synthetic incidence data (gh#275 Phase 2). Authors a small SIR with a
//! `log_normal` prior on `beta` (nuts requires a proper prior), simulates
//! observations at the true `beta` via the ODE backend, runs a `nuts` fit from a
//! wrong start, and checks the posterior mean recovers the truth with no
//! divergences — the CLI analogue of `ode_nuts::tests::ode_nuts_recovers_known_beta`.

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
    let base = std::env::temp_dir().join(format!("camdl_nutsode_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

const TRUE_BETA: f64 = 0.9;
const START_BETA: f64 = 0.4;
const WARMUP: usize = 150;

const MODEL: &str = r#"
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
    projected     = incidence(infection)
    emit_schedule = every 3 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;

fn beta_samples(trace: &Path, drop: usize) -> Vec<f64> {
    let text = std::fs::read_to_string(trace).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("trace header").split('\t').collect();
    let beta_col = header.iter().position(|h| *h == "beta").expect("beta column");
    lines.skip(drop)
        .filter_map(|l| l.split('\t').nth(beta_col).and_then(|c| c.parse::<f64>().ok()))
        .collect()
}

fn find_traces(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() { stack.push(p); }
                else if p.file_name().and_then(|n| n.to_str()) == Some("trace.tsv") { out.push(p); }
            }
        }
    }
    out
}

#[test]
fn nuts_ode_recovers_known_beta_from_incidence() {
    let bin = camdl_bin();
    if !bin.exists() || camdlc().is_none() {
        eprintln!("skip: release camdl / camdlc.exe missing (run `make build`)");
        return;
    }
    let tmp = tempdir("recovery");
    let model_path = tmp.path().join("sir.camdl");
    std::fs::write(&model_path, MODEL).unwrap();
    let ir = tmp.path().join("sir.ir.json");
    let out = Command::new(camdlc().unwrap()).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();

    // Synthetic data at the true beta.
    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, format!("beta = {TRUE_BETA}\ngamma = 0.3\nN0 = 10000\n")).unwrap();
    let data = tmp.path().join("cases.tsv");
    let sim = Command::new(&bin)
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"]).arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));

    let out_dir = tmp.path().join("out");
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

[stages.nuts]
algorithm = "nuts"
backend = "ode"
chains = 2
warmup = {WARMUP}
samples = 250
# Controlled recovery from the declared start (like mh_ode_recovery): the default
# init is UniformUnconstrained (dispersed), which is right for a real fit but adds
# variance a short-warmup recovery check shouldn't depend on.
init = "single"
"#, out = out_dir.display(), ir = ir.display(), data = data.display())).unwrap();

    let status = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit_toml)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status().unwrap();
    assert!(status.success(), "nuts+ode `fit run` must succeed (exit 0)");

    let traces = find_traces(&out_dir);
    assert!(!traces.is_empty(), "no chain trace.tsv produced under {}", out_dir.display());
    let betas: Vec<f64> = traces.iter().flat_map(|t| beta_samples(t, 50)).collect();
    assert!(betas.len() >= 100, "too few post-warmup beta samples ({})", betas.len());
    let mean = betas.iter().sum::<f64>() / betas.len() as f64;
    eprintln!("nuts+ode: posterior mean beta = {mean:.4} (true {TRUE_BETA}), n={}", betas.len());
    assert!(
        (mean - TRUE_BETA).abs() < 0.1,
        "nuts+ode did not recover beta: posterior mean {mean:.4}, true {TRUE_BETA}"
    );

    // gh#275: nuts is a first-class Bayesian cell — it must render in
    // `fit table` / `fit summary` with ESS/iteration, not error with "unknown
    // fit-stage method `nuts`". This is the cross-cutting matrix check that the
    // runner's `nuts_summary.json` + `draws.tsv` and the `MethodResult::Nuts`
    // loader are wired end-to-end. (Before this, nuts wrote no posterior
    // summary and had no loader arm, so both commands dropped the stage.)
    let fits_dir = out_dir.join("fits");
    let table = Command::new(&bin)
        .args(["fit", "table"]).arg(&fits_dir)
        .args(["--format", "csv"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(
        table.status.success(),
        "fit table on a nuts fit must succeed: {}",
        String::from_utf8_lossy(&table.stderr)
    );
    let table_txt = String::from_utf8_lossy(&table.stdout);
    let header = table_txt.lines().next().expect("csv header");
    let epi_col = header.split(',').position(|c| c == "ess_per_iter")
        .expect("fit table csv must carry an ess_per_iter column");
    let nuts_has_ess = table_txt.lines().skip(1).any(|row| {
        row.split(',').nth(epi_col)
            .and_then(|c| c.parse::<f64>().ok())
            .map(|v| v > 0.0)
            .unwrap_or(false)
    });
    assert!(nuts_has_ess,
        "the nuts row must report a positive ess_per_iter off its posterior:\n{table_txt}");

    // The single fit dir under fits/ (there is exactly one fit here).
    let fit_dir = std::fs::read_dir(&fits_dir).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
        .expect("a fit dir under fits/");
    let summary = Command::new(&bin)
        .args(["fit", "summary"]).arg(&fit_dir)
        .args(["--format", "text", "--no-color"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(
        summary.status.success(),
        "fit summary on a nuts fit must succeed: {}",
        String::from_utf8_lossy(&summary.stderr)
    );
    let summary_txt = String::from_utf8_lossy(&summary.stdout);
    assert!(summary_txt.contains("ESS/iter"),
        "nuts fit summary must report ESS/iter off the posterior:\n{summary_txt}");
    assert!(summary_txt.contains(" · nuts · "),
        "the stage's identity line must name the sampler that produced \
         it:\n{summary_txt}");

    // The `RhatHigh` findings are pushed as a side effect of RENDERING the
    // convergence block, and nuts never rendered one — so `diagnostics.json`
    // was never written for a `nuts` stage and the diagnostics surface was
    // blind to non-convergence on every ODE+NUTS fit, however badly the chains
    // disagreed. The file existing is the check; its CONTENTS are a property of
    // this (converged) fixture and are not asserted.
    let stage_dir = find_stage_dir(&fit_dir, "nuts_summary.json")
        .expect("the nuts stage dir carries its summary");
    assert!(
        stage_dir.join("diagnostics.json").exists(),
        "a nuts stage must persist diagnostics.json like pgas and pmmh do; \
         stage dir holds: {:?}",
        std::fs::read_dir(&stage_dir).unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect::<Vec<_>>()
    );
}

/// The directory under `fit_dir` holding `marker`.
fn find_stage_dir(fit_dir: &Path, marker: &str) -> Option<PathBuf> {
    let mut stack = vec![fit_dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join(marker).exists() {
            return Some(d);
        }
        for e in std::fs::read_dir(&d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            }
        }
    }
    None
}
