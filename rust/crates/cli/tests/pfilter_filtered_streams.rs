//! `camdl pfilter --save-filtered-streams`: each stream's conditional mean
//! `E[y_t | x_t, θ]` summarised over the particles before (`predictive`) and
//! after (`filtered`) each step is scored.
//!
//! Three properties, end to end through the release binary:
//!
//! - On a model whose filtering distribution is known exactly, the summaries
//!   match it: `A₀ ~ Poisson(λ)`, every individual moves `A → B` on the first
//!   day (per-capita rate `fast` = 1000 per day, so the chain-binomial probability
//!   `1 − exp(−1000)` is 1 in floating point), and the daily flow is observed
//!   as `y ~ Poisson(ρ · flow)`. On day 1 the stream mean is `ρ A₀`: its
//!   predictive law is `ρ · Poisson(λ)`, and its filtered law is
//!   `p(A₀ | y₁) ∝ Poisson(A₀; λ) · Poisson(y₁; ρ A₀)`, computed here by
//!   summation. On later days the flow is 0 on every particle, a point mass.
//! - The `predictive` median and mean are the one-step `state_q50` and
//!   `state_mean` that `--trace` already reports, row by row.
//! - The flag changes nothing else: the log-likelihood and every other output
//!   file are byte-identical with and without it. (Run identity is the
//!   `presentation_save_filtered_streams` case in `pfilter_input_classes.rs`.)
//!
//! Skipped when the release `camdl` or `camdlc.exe` is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    p.exists().then_some(p)
}

fn compile(dir: &Path, name: &str, src: &str) -> Option<PathBuf> {
    let camdlc = camdlc()?;
    let model = dir.join(format!("{name}.camdl"));
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join(format!("{name}.ir.json"));
    std::fs::write(&ir, &out.stdout).unwrap();
    Some(ir)
}

fn pfilter(dir: &Path, args: &[&str]) -> std::process::Output {
    let out = Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .arg("pfilter").args(args).args(["--no-store", "--no-progress"])
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(out.status.success(), "pfilter {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    out
}

/// One row of the file: (time, stream, phase, [mean, q05, q25, q50, q75, q95]).
type Row = (f64, String, String, [f64; 6]);

fn read_rows(path: &Path) -> Vec<Row> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("time\tstream\tphase\tmean\tq05\tq25\tq50\tq75\tq95"));
    lines.map(|l| {
        let c: Vec<&str> = l.split('\t').collect();
        assert_eq!(c.len(), 9, "row: {l}");
        let v: Vec<f64> = c[3..].iter().map(|x| x.parse().unwrap()).collect();
        (c[0].parse().unwrap(), c[1].to_string(), c[2].to_string(), v.try_into().unwrap())
    }).collect()
}

struct Tmp(PathBuf);
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tmp(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("camdl_fstreams_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

// ─── The exact case ─────────────────────────────────────────────────────────

const LAMBDA: f64 = 20.0;
const RHO: f64 = 0.5;
const Y1: u64 = 7;

const EXACT_SRC: &str = r#"
time_unit = 'days
compartments { A, B }
parameters {
  lambda : count       in [1, 1000]
  rho    : probability in [0.01, 1.0]
  fast   : rate        in [1, 10000]
}
transitions {
  move : A --> B @ fast * A
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(move)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}
init { A ~ poisson(rate = lambda)  B = 0 }
simulate { from = 0 'days  to = 3 'days  dt = 1 'days }
"#;

fn ln_poisson(k: u64, mu: f64) -> f64 {
    if mu == 0.0 { return if k == 0 { 0.0 } else { f64::NEG_INFINITY }; }
    k as f64 * mu.ln() - mu - (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
}

/// The exact law of the day-1 stream mean `ρ A₀` as (value, probability)
/// pairs over `A₀ = 0..200`: the prior, or the posterior given `y₁`.
fn day1_law(posterior: bool) -> Vec<(f64, f64)> {
    let lw: Vec<f64> = (0..200u64).map(|a| {
        ln_poisson(a, LAMBDA) + if posterior { ln_poisson(Y1, RHO * a as f64) } else { 0.0 }
    }).collect();
    let max = lw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = lw.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = w.iter().sum();
    (0..200u64).map(|a| (RHO * a as f64, w[a as usize] / z)).collect()
}

fn cdf(law: &[(f64, f64)], x: f64) -> f64 {
    law.iter().filter(|(v, _)| *v <= x + 1e-9).map(|(_, p)| p).sum()
}

/// `got` is a `p`-quantile of `law` up to `tol` in probability: the
/// cumulative probability reaches `p − tol` at `got` and has not passed
/// `p + tol` just below it (one atom down, `ρ`).
fn assert_quantile(law: &[(f64, f64)], p: f64, got: f64, what: &str) {
    let tol = 0.03;
    let at = cdf(law, got);
    let below = cdf(law, got - RHO);
    assert!(at >= p - tol && below <= p + tol,
        "{what}: q{p} = {got}, exact CDF there {at:.4} and one atom below {below:.4}");
}

#[test]
fn summaries_match_the_exact_filtering_distribution() {
    let t = tmp("exact");
    if !camdl_bin().exists() { return; }
    let Some(ir) = compile(&t.0, "exact", EXACT_SRC) else { return };
    std::fs::write(t.0.join("p.toml"), format!("lambda = {LAMBDA}\nrho = {RHO}\nfast = 1000\n")).unwrap();
    std::fs::write(t.0.join("d.tsv"), format!("time\tcases\n1\t{Y1}\n2\t0\n3\t0\n")).unwrap();
    pfilter(&t.0, &[ir.to_str().unwrap(), "--params", "p.toml", "--data", "d.tsv",
        "--particles", "20000", "--seed", "3", "--save-filtered-streams", "s.tsv"]);
    let rows = read_rows(&t.0.join("s.tsv"));

    // Three days × one stream × two phases, in that order.
    let keys: Vec<(f64, &str, &str)> = rows.iter().map(|r| (r.0, r.1.as_str(), r.2.as_str())).collect();
    assert_eq!(keys, vec![
        (1.0, "cases", "predictive"), (1.0, "cases", "filtered"),
        (2.0, "cases", "predictive"), (2.0, "cases", "filtered"),
        (3.0, "cases", "predictive"), (3.0, "cases", "filtered"),
    ]);

    for (row, posterior) in [(&rows[0], false), (&rows[1], true)] {
        let law = day1_law(posterior);
        let mean: f64 = law.iter().map(|(v, p)| v * p).sum();
        let sd = law.iter().map(|(v, p)| p * (v - mean).powi(2)).sum::<f64>().sqrt();
        // Monte Carlo error of a weighted mean over ≥ 2000 effective particles.
        let what = &row.2;
        assert!((row.3[0] - mean).abs() < 4.0 * sd / 2000f64.sqrt(),
            "{what} mean {} vs exact {mean:.4} (sd {sd:.3})", row.3[0]);
        for (k, p) in [0.05, 0.25, 0.50, 0.75, 0.95].into_iter().enumerate() {
            assert_quantile(&law, p, row.3[k + 1], what);
        }
    }
    // The posterior moved: y₁ = 7 sits well below the prior mean ρλ = 10.
    assert!(rows[1].3[3] < rows[0].3[3], "filtered median should fall below the predictive one");

    // Days 2 and 3: everyone moved on day 1, so the flow is 0 on every particle.
    for r in &rows[2..] {
        assert_eq!(r.3, [0.0; 6], "{r:?}");
    }
}

// ─── Agreement with --trace, and nothing else changes ───────────────────────

const SIR_SRC: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 5.0]
  gamma : rate        in [0.01, 1.0]
  rho   : probability in [0.01, 1.0]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 2 'days)
    projected     = incidence(recovery)
    emit_schedule = every 2 'days
    cases ~ neg_binomial(mean = rho * projected, r = 10.0)
  }
}
init { S = 990  I = 10  R = 0 }
simulate { from = 0 'days  to = 20 'days  dt = 1 'days }
"#;

fn sir(dir: &Path) -> Option<PathBuf> {
    let ir = compile(dir, "sir", SIR_SRC)?;
    std::fs::write(dir.join("p.toml"), "beta = 0.6\ngamma = 0.2\nrho = 0.5\n").unwrap();
    std::fs::write(dir.join("d.tsv"),
        "time\tcases\n2\t2\n4\t4\n6\t7\n8\t9\n10\t15\n12\t20\n14\t31\n16\t40\n18\t52\n20\t60\n").unwrap();
    Some(ir)
}

#[test]
fn predictive_median_is_the_trace_one_step_median() {
    let t = tmp("trace");
    if !camdl_bin().exists() { return; }
    let Some(ir) = sir(&t.0) else { return };
    pfilter(&t.0, &[ir.to_str().unwrap(), "--params", "p.toml", "--data", "d.tsv",
        "--particles", "500", "--seed", "11", "--trace", "trace.tsv",
        "--save-filtered-streams", "s.tsv"]);
    let predictive: Vec<Row> = read_rows(&t.0.join("s.tsv")).into_iter()
        .filter(|r| r.2 == "predictive").collect();
    let trace = std::fs::read_to_string(t.0.join("trace.tsv")).unwrap();
    let mut lines = trace.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let trace_rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert_eq!(trace_rows.len(), predictive.len());
    assert_eq!(trace_rows.len(), 10);
    for (tr, r) in trace_rows.iter().zip(&predictive) {
        assert_eq!(tr[col("time")].parse::<f64>().unwrap(), r.0);
        // The trace prints the median rounded to an integer and the mean to
        // one decimal; the same formatting of this file's values matches.
        assert_eq!(tr[col("state_q50")], format!("{:.0}", r.3[3]), "t = {}", r.0);
        assert_eq!(tr[col("state_mean")], format!("{:.1}", r.3[0]), "t = {}", r.0);
    }
}

#[test]
fn the_flag_changes_no_other_output() {
    let t = tmp("same");
    if !camdl_bin().exists() { return; }
    let Some(ir) = sir(&t.0) else { return };
    let run = |tag: &str, extra: &[&str]| -> Vec<(String, Vec<u8>)> {
        let d = t.0.join(tag);
        std::fs::create_dir_all(&d).unwrap();
        for f in ["p.toml", "d.tsv"] { std::fs::copy(t.0.join(f), d.join(f)).unwrap(); }
        let mut args = vec![ir.to_str().unwrap(), "--params", "p.toml", "--data", "d.tsv",
            "--particles", "300", "--seed", "4", "--trace", "trace.tsv",
            "--pf-health", "health.tsv", "--save-paths", "paths.tsv", "--n-paths", "5",
            "--save-filtering", "filtering.tsv", "--save-prequential", "preq",
            "--save-final-state", "final.tsv"];
        args.extend_from_slice(extra);
        let out = pfilter(&d, &args);
        let mut files = vec![("stdout".to_string(), out.stdout)];
        for f in ["trace.tsv", "health.tsv", "paths.tsv", "filtering.tsv", "preq.tsv",
                  "preq.streams.tsv", "preq.json", "final.tsv"] {
            files.push((f.to_string(), std::fs::read(d.join(f)).unwrap()));
        }
        files
    };
    let without = run("without", &[]);
    let with = run("with", &["--save-filtered-streams", "s.tsv"]);
    assert!(t.0.join("with/s.tsv").exists());
    for ((name, a), (_, b)) in without.iter().zip(&with) {
        assert!(a == b, "{name} differs with --save-filtered-streams");
    }
}

#[test]
fn one_row_per_scheduled_stream() {
    let t = tmp("multi");
    if !camdl_bin().exists() { return; }
    let src = SIR_SRC.replace("r = 10.0)\n  }\n", r#"r = 10.0)
  }
  prevalence {
    columns       { time : time, prevalence : count }
    projected     = I
    emit_schedule = every 4 'days
    prevalence ~ poisson(rate = projected)
  }
"#);
    assert!(src.contains("prevalence {"));
    let Some(ir) = compile(&t.0, "multi", &src) else { return };
    let Some(_) = sir(&t.0) else { return };
    std::fs::write(t.0.join("prev.tsv"), "time\tprevalence\n4\t20\n8\t40\n12\t80\n16\t150\n20\t240\n").unwrap();
    pfilter(&t.0, &[ir.to_str().unwrap(), "--params", "p.toml",
        "--data", "cases=d.tsv", "--data", "prevalence=prev.tsv",
        "--particles", "300", "--seed", "2", "--save-filtered-streams", "s.tsv"]);
    let rows = read_rows(&t.0.join("s.tsv"));
    let times = |stream: &str| -> Vec<f64> {
        rows.iter().filter(|r| r.1 == stream && r.2 == "filtered").map(|r| r.0).collect()
    };
    assert_eq!(times("cases"), (1..=10).map(|k| 2.0 * k as f64).collect::<Vec<_>>());
    assert_eq!(times("prevalence"), (1..=5).map(|k| 4.0 * k as f64).collect::<Vec<_>>());
    assert_eq!(rows.len(), 2 * (10 + 5));
}
