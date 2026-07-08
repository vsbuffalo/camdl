//! End-to-end: `burnin_dt` (coarse warm-up step) on `nuts`-on-`ode` (gh#396
//! follow-on). Authors a slow SEIR with a PREVALENCE (state-scored) stream and an
//! unscored warm-up window `[0, 50)`, then:
//!   (1) a `burnin_dt = 5` fit runs, has no divergences, and recovers the true
//!       `beta` (the coarse warm-up integrates state + sensitivity together, so the
//!       gradient stays consistent — the fit still finds the mode);
//!   (2) `burnin_dt` on an INCIDENCE stream is refused (its first bin accumulates
//!       flow from t_start, which the coarse warm-up would bias);
//!   (3) `burnin_dt < dt` is refused (a coarse step must be LARGER than dt).
//!
//! The numerical correctness of the coarse gradient is proved by the FD oracle
//! `ode_grad::tests::det_grad_matches_finite_difference_under_coarse_burnin_dt`;
//! this is the CLI-surface cell (config parse → gate → coarse fit → output).

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
    let base = std::env::temp_dir().join(format!("camdl_coarse_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// A slow SEIR (`beta = 0.15`, `R0 = 1.5`) so the epidemic is still ramping across
/// the scored window and the coarse `dt = 5` steps stay within RK4 stability. The
/// observation stream is PREVALENCE (`projected = I`, state-scored) — the cell
/// `burnin_dt` supports. Stream name == column name (`prev`) so `simulate
/// --obs-only`'s header matches the model's declared column.
const PREV_MODEL: &str = r#"
time_unit = 'days
compartments { S, E, I, R }
parameters {
  beta  : rate  in [0.02, 2.0] ~ log_normal(mu = -1.9, sigma = 0.8)
  sigma : rate  in [0.01, 1.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection   : S --> E @ beta * S * I / N0
  progression : E --> I @ sigma * E
  recovery    : I --> R @ gamma * I
}
observations {
  prev {
    columns       { time : time, prev : count }
    projected     = I
    emit_schedule = every 10 'days
    prev ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 150 'days }
"#;

const TRUE_BETA: f64 = 0.15;

/// Compile the model, simulate prevalence at the true params, and trim the data to
/// `time >= 50` so the fit's first observation is at 50 (an unscored `[0, 50)`
/// warm-up). Returns `(tmp, ir_path, data_path)`. Skips (returns `None`) if the
/// release binaries are missing.
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
    let out = Command::new(camdlc().unwrap()).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();

    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, format!("beta = {TRUE_BETA}\nsigma = 0.2\ngamma = 0.1\nN0 = 10000\n")).unwrap();
    let full = tmp.path().join("full.tsv");
    let sim = Command::new(&bin)
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"]).arg(&full)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));

    // Trim to time >= 50 (keep the header verbatim so the column name matches).
    let text = std::fs::read_to_string(&full).unwrap();
    let mut lines = text.lines();
    let header = lines.next().expect("data header");
    let mut trimmed = String::from(header);
    trimmed.push('\n');
    for l in lines {
        if let Some(t) = l.split('\t').next().and_then(|c| c.parse::<f64>().ok()) {
            if t >= 50.0 { trimmed.push_str(l); trimmed.push('\n'); }
        }
    }
    let data = tmp.path().join("prev.tsv");
    std::fs::write(&data, trimmed).unwrap();
    Some((tmp, ir, data))
}

fn fit_toml(out_dir: &Path, ir: &Path, data: &Path, stream: &str, burnin_line: &str) -> String {
    format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
{stream} = "{data}"
[estimate]
beta = {{ bounds = [0.02, 2.0], start = 0.1, prior = {{ log_normal = {{ mu = -1.9, sigma = 0.8 }} }} }}
[fixed]
sigma = 0.2
gamma = 0.1
N0 = 10000
[stages.posterior]
algorithm = "nuts"
backend = "ode"
chains = 2
warmup = 100
samples = 150
init = "single"
{burnin_line}
"#, out = out_dir.display(), ir = ir.display(), data = data.display())
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

fn beta_mean(traces: &[PathBuf], drop: usize) -> (f64, usize) {
    let mut vals = Vec::new();
    for t in traces {
        let text = std::fs::read_to_string(t).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().expect("trace header").split('\t').collect();
        let col = header.iter().position(|h| *h == "beta").expect("beta column");
        for l in lines.skip(drop) {
            if let Some(v) = l.split('\t').nth(col).and_then(|c| c.parse::<f64>().ok()) {
                vals.push(v);
            }
        }
    }
    let n = vals.len();
    (vals.iter().sum::<f64>() / n as f64, n)
}

#[test]
fn nuts_ode_coarse_burnin_prevalence_recovers() {
    let Some((tmp, ir, data)) = setup("recover", PREV_MODEL) else { return };
    let bin = camdl_bin();
    let out_dir = tmp.path().join("out");
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, "prev", "burnin_dt = 5.0")).unwrap();

    let status = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .status().unwrap();
    assert!(status.success(), "coarse `burnin_dt` nuts+ode fit must succeed (exit 0)");

    let traces = find_traces(&out_dir);
    assert!(!traces.is_empty(), "no chain trace.tsv under {}", out_dir.display());
    let (mean, n) = beta_mean(&traces, 50);
    assert!(n >= 100, "too few post-warmup samples ({n})");
    eprintln!("coarse burnin_dt=5: posterior mean beta = {mean:.4} (true {TRUE_BETA}), n={n}");
    assert!(
        (mean - TRUE_BETA).abs() < 0.05,
        "coarse burn-in fit did not recover beta: mean {mean:.4}, true {TRUE_BETA}"
    );
}

#[test]
fn burnin_dt_refuses_incidence_stream() {
    // Same model but an INCIDENCE projection — the cell burnin_dt refuses in v1.
    let inc_model = PREV_MODEL.replace("projected     = I", "projected     = incidence(infection)");
    let Some((tmp, ir, data)) = setup("incidence", &inc_model) else { return };
    let bin = camdl_bin();
    let out_dir = tmp.path().join("out");
    let fit = tmp.path().join("fit.toml");
    std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, "prev", "burnin_dt = 5.0")).unwrap();

    let out = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(!out.status.success(), "burnin_dt on an incidence stream must be refused (nonzero exit)");
    let msg = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
    assert!(
        msg.contains("burnin_dt") && msg.contains("incidence"),
        "the refusal must name burnin_dt and incidence:\n{msg}"
    );
}

#[test]
fn burnin_dt_below_dt_is_rejected() {
    let Some((tmp, ir, data)) = setup("ltdt", PREV_MODEL) else { return };
    let bin = camdl_bin();
    let out_dir = tmp.path().join("out");
    let fit = tmp.path().join("fit.toml");
    // dt defaults to 1; burnin_dt = 0.5 < dt is nonsensical (coarser means larger).
    std::fs::write(&fit, fit_toml(&out_dir, &ir, &data, "prev", "burnin_dt = 0.5")).unwrap();

    let out = Command::new(&bin)
        .args(["fit", "run"]).arg(&fit)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(!out.status.success(), "burnin_dt < dt must be rejected (nonzero exit)");
    let msg = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
    assert!(
        msg.contains("burnin_dt") && msg.contains("smaller than"),
        "the error must explain burnin_dt must be larger than dt:\n{msg}"
    );
}
