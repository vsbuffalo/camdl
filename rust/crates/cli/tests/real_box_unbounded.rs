//! A `real` or `instant` parameter that declares no range and is given no
//! fit-config bounds searches the whole line, `(−∞, ∞)`. Nothing would keep
//! it in a narrower box: with an infinite end its transform is the identity,
//! which clamps nothing. So the record says `search: ["-inf", "inf"]`, and
//! `from_prior` draws it from its prior, negative values included.
//!
//! A positive kind with no bounds keeps `[0, ∞)`. Either way the box has an
//! infinite end, so the rules that draw within it (`uniform`, `lhs`,
//! `uniform_unconstrained`, at two or more chains) and `survey`'s
//! Latin-hypercube landscape refuse the parameter rather than substitute a
//! jitter around its start.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

/// SIR with `delta : real` and `eta : rate`, declaring no range, which the
/// dynamics ignore.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate        in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate        in [0.01, 0.5]
  N0    : count
  I0    : count
  delta : real        ~ normal(mu = -1.0, sigma = 1.0)
  eta   : rate        ~ log_normal(mu = 0.0, sigma = 1.0)
}
let N = S + I + R
transitions {
  infection : S --> I  @ (beta + 0.0 * delta + 0.0 * eta) * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = 0.6 * projected, r = 10.0)
  }
}
simulate { from = 0 'days  to = 56 'days }
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// Run `camdl fit run` with `delta` estimated under `method`, plus `extra`
/// arguments. Returns the output and the results directory.
fn fit(method: &str, extra: &[&str]) -> (Output, tempfile::TempDir) {
    let mut args = vec!["fit", "run", "fit.toml", "--seed", "1"];
    args.extend_from_slice(extra);
    run("delta", method, &args)
}

/// Run `camdl <args>` in a directory holding the model, the data and a
/// fit.toml estimating `free` (with no bounds and an explicit `rw_sd`); the
/// other unbounded parameter is held.
fn run(free: &str, method: &str, args: &[&str]) -> (Output, tempfile::TempDir) {
    let held = if free == "delta" { "eta = 0.1" } else { "delta = 0.0" };
    let toml = format!(r#"[model]
camdl = "model.camdl"
[data.observations]
weekly_cases = "weekly_cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.25 }}
{free} = {{ start = 0.5, rw_sd = 0.1 }}
[fixed]
gamma = 0.15
N0 = 10000
I0 = 10
{held}
[method]
{method}
"#);
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), toml).unwrap();
    let out = Command::new(binary())
        .args(args)
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("res"))
        .current_dir(tmp.path())
        .output()
        .expect("spawn");
    (out, tmp)
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Every file named `name` under `root`.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|f| f == name) {
                found.push(p);
            }
        }
    }
    found
}

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\n\
                   particles = 50\niterations = 2\ncooling = 0.7";
const PMMH_4: &str = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 4\n\
                      particles = 20\niterations = 4\nburn_in = 1";

#[test]
fn an_unbounded_real_parameter_records_the_whole_line_as_its_box() {
    let (out, tmp) = fit(IF2, &[]);
    assert!(out.status.success(), "{}", stderr(&out));
    let metas = find_all(tmp.path(), "fit.meta.json");
    assert_eq!(metas.len(), 1, "{metas:?}");
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(&metas[0]).unwrap()).unwrap();
    let delta = &meta["partition"]["parameters"]["delta"];
    assert_eq!(delta["search"], serde_json::json!(["-inf", "inf"]), "{delta:#}");
    assert_eq!(delta["search_from"], "support", "{delta:#}");
    assert_eq!(delta["transform"], "identity", "{delta:#}");
}

#[test]
fn a_box_spreading_start_rule_refuses_it() {
    for rule in ["uniform", "lhs", "uniform_unconstrained"] {
        let (out, _tmp) = fit(PMMH_4, &["--starts", rule]);
        let err = stderr(&out);
        assert!(!out.status.success(), "{rule}: expected a refusal:\n{err}");
        assert!(err.contains("these parameters have no finite one: `delta` [-inf, inf].")
            && err.contains("bounds = [lo, hi]") && err.contains("`from_prior`"),
            "{rule}: the refusal names the parameter, its box and the remedies:\n{err}");
    }
}

#[test]
fn from_prior_starts_it_below_zero() {
    let (out, tmp) = fit(PMMH_4, &["--starts", "from_prior"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let files = find_all(tmp.path(), "chain_starts.tsv");
    assert_eq!(files.len(), 1, "{files:?}");
    let text = std::fs::read_to_string(&files[0]).unwrap();
    let mut rows = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = rows.next().unwrap().split('\t').collect();
    let i = header.iter().position(|h| *h == "delta").expect("delta column");
    let status = header.iter().position(|h| *h == "status").expect("status column");
    // Every draw, the ones the filter could not score and redrew included.
    let starts: Vec<f64> = rows.map(|l| l.split('\t').nth(i).unwrap().parse().unwrap()).collect();
    let accepted = text.lines().filter(|l| !l.starts_with('#')).skip(1)
        .filter(|l| l.split('\t').nth(status) == Some("accepted")).count();
    assert_eq!(accepted, 4, "{text}");
    // `normal(-1, 1)` puts 84% of its mass below zero; four starts all at or
    // above it would have probability 0.16^4 ≈ 7e-4 at this seed's draw.
    assert!(starts.iter().any(|v| *v < 0.0), "no start below zero: {starts:?}");
}

#[test]
fn a_box_spreading_start_rule_refuses_an_unbounded_rate_too() {
    for rule in ["uniform", "lhs", "uniform_unconstrained"] {
        let (out, _tmp) = run("eta", PMMH_4, &["fit", "run", "fit.toml", "--seed", "1",
            "--starts", rule]);
        let err = stderr(&out);
        assert!(!out.status.success(), "{rule}: expected a refusal:\n{err}");
        assert!(err.contains("these parameters have no finite one: `eta` [0, inf]."),
            "{rule}: the refusal names the parameter and its box:\n{err}");
    }
    // The rule that does not draw within the box runs.
    let (out, _tmp) = run("eta", PMMH_4, &["fit", "run", "fit.toml", "--seed", "1",
        "--starts", "from_prior"]);
    assert!(out.status.success(), "{}", stderr(&out));
}

#[test]
fn survey_refuses_a_parameter_with_no_finite_box() {
    for (free, shown) in [("delta", "`delta` [-inf, inf]"), ("eta", "`eta` [0, inf]")] {
        let (out, _tmp) = run(free, IF2, &["survey", "model.camdl", "--fit", "fit.toml",
            "--n-points", "4", "--eval-particles", "20", "--output", "srv"]);
        let err = stderr(&out);
        assert!(!out.status.success(), "{free}: expected a refusal:\n{err}");
        assert!(err.contains(&format!("these parameters have no finite one: {shown}."))
            && err.contains("`bounds = [lo, hi]`"),
            "{free}: the refusal names the parameter, its box and the remedy:\n{err}");
    }
}
