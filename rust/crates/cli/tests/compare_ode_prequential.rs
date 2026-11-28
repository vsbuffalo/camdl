//! gh#312: `camdl compare` on a fit that ran `mh` on the `ode` backend.
//!
//! An ODE + MH fit of a small SIR, sealed at `holdout_after = 42`, is scored
//! free-forward, the exact observation density at each draw's trajectory,
//! which does not depend on the seed: compare predicts it once, and its row's
//! elpd is the sum of the stored score's log scores. A particle filter's trace
//! of the same data is a Monte-Carlo estimate, and the two are refused as a
//! pair. That an ODE score equals the observation density at the trajectory
//! is `camdl score`'s own test.
//!
//! Shells out to the release `camdl`; the `.camdl` model is compiled by the
//! `camdlc` on PATH (`make test-rust` puts this checkout's first).

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const RHO: f64 = 0.6;
const K: f64 = 10.0;

/// The fit config; `estimate_gamma` is the one structural difference between
/// the two compared models. `holdout` false gives the design config the
/// replay reads (the same data, nothing withheld).
fn fit_toml(estimate_gamma: bool, holdout: bool) -> String {
    let gamma = if estimate_gamma {
        "gamma = { bounds = [0.01, 0.5], start = 0.15 }\n"
    } else { "" };
    let fixed_gamma = if estimate_gamma { "" } else { "gamma = 0.2\n" };
    let holdout = if holdout { "holdout_after = 42\n" } else { "" };
    format!(r#"store = "../store"
[model]
camdl = "model.camdl"
[data]
{holdout}[data.observations]
weekly_cases = "cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
{gamma}[fixed]
{fixed_gamma}N0  = 10000
I0  = 10
rho = {RHO}
k   = {K}
[method]
algorithm = "mh"
backend = "ode"
chains = 1
iterations = 300
burn_in = 100
starts = "single"
"#)
}

/// The store every fit of a test is stored in, `<root>/store`, for `dir`
/// either the test's root or one fit's directory `<root>/<name>`, which
/// holds its `fit.toml`.
fn store_for(dir: &Path) -> PathBuf {
    let root = if dir.join("fit.toml").is_file() { dir.parent().unwrap() } else { dir };
    root.join("store")
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", store_for(dir))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .current_dir(dir)
        .output().expect("spawn camdl")
}

fn ok(out: &Output, what: &str) -> String {
    assert!(out.status.success(), "{what} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Fit one model in `root/name`, stored in `root/store` under the label
/// `name`, and return its directory.
fn fit(root: &Path, name: &str, estimate_gamma: bool) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), fit_toml(estimate_gamma, true)).unwrap();
    std::fs::write(dir.join("design.toml"), fit_toml(estimate_gamma, false)).unwrap();
    ok(&camdl(&dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", name]), "fit run");
    dir
}

/// Two ODE fits compared in-sample, with replicates by default: an ODE
/// free-forward score is the same at every seed, so each is predicted and
/// scored once, its row records one replicate and no filter-noise SE, the
/// table prints none, and its elpd is the sum of its stored score's log
/// scores over the training rows.
#[test]
fn an_ode_free_forward_score_is_predicted_once_and_is_its_stored_sum() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fit(root, "m0", true);
    fit(root, "m1", false);
    let line = ["compare", "@m0", "@m1", "--in-sample", "--draws", "4"];
    let out = camdl(root, &[&line[..], &["--format", "json"]].concat());
    let v: serde_json::Value = serde_json::from_str(&ok(&out, "compare")).unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(err.matches("does not depend on the seed").count(), 2, "{err}");
    let scores = fit_store_support::folders_of(&store_for(root), "score");
    assert_eq!(scores.len(), 2, "one score per fit: {scores:?}");
    let mut sums: Vec<f64> = scores.iter().map(|s| {
        let text = std::fs::read_to_string(s.join("scored_rows.tsv")).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let col = |n: &str| header.iter().position(|h| *h == n).unwrap();
        lines.map(|l| l.split('\t').collect::<Vec<_>>())
            .filter(|c| c[col("mode")] == "free_forward" && c[col("role")] == "trained")
            .map(|c| c[col("log_score")].parse::<f64>().unwrap())
            .sum()
    }).collect();
    let mut elpds: Vec<f64> = Vec::new();
    for r in v["rows"].as_array().unwrap() {
        assert!(r["basis"]["ode_free_forward"].is_object(), "{r}");
        let d = &r["derivation"];
        assert_eq!((d["particles"].as_u64(), d["replicates"].as_u64()), (Some(0), Some(1)),
            "no filter ran, and a deterministic score has no replicates: {d}");
        assert!(d.get("mc_se_elpd").is_none() && r["mc_se_elpd"].is_null(), "and so no filter-noise SE: {r}");
        elpds.push(r["elpd"].as_f64().unwrap());
    }
    sums.sort_by(f64::total_cmp);
    elpds.sort_by(f64::total_cmp);
    assert!(sums.iter().zip(&elpds).all(|(a, b)| (a - b).abs() <= 1e-9 * a.abs().max(1.0)),
        "each elpd is its stored score's sum: {elpds:?} vs {sums:?}");
    let table = ok(&camdl(root, &line), "compare table");
    assert!(!table.contains("±0.00") && !table.contains("filter-noise MC SE"), "{table}");
}

/// A filtered trace (a particle filter's estimate) and an ODE trace (exact) are
/// refused as a pair: the Δ would carry the filter's log bias as a model
/// difference. No flag waives it.
#[test]
fn an_ode_trace_is_not_differenced_against_a_filtered_one() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let m0 = fit(root, "m0", true);
    std::fs::write(m0.join("theta.toml"),
        "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n").unwrap();
    ok(&camdl(&m0, &["pfilter", "model.camdl", "--params", "theta.toml",
        "--data", "weekly_cases=cases.tsv", "--particles", "100",
        "--save-prequential", "pf", "--no-store"]), "pfilter --save-prequential");

    // Both in-sample, so the conditioning check passes and the basis check is
    // the one that refuses.
    let out = camdl(root, &["compare", "@m0", "m0/pf.json", "--in-sample",
        "--draws", "2", "--replicates", "1", "--allow-mixed-provenance",
        "--allow-mismatched-filter-settings", "--allow-data-mismatch"]);
    assert_eq!(out.status.code(), Some(2), "refused as incommensurable:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("mix predictive bases"), "{err}");
    assert!(err.contains("ode_free_forward") && err.contains("filtered"), "{err}");
}
