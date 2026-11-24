//! `camdl fit summary --point`: a sampler run's posterior summarised per
//! parameter, as a params file whose `[provenance]` block `simulate --params`
//! and `predict MODEL --params` read back.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

mod fit_store_support;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let bin = Path::new(&manifest).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    bin
}

/// A closed SIR with a weekly negative-binomial observation.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 56 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The fit file: `beta` and `gamma` estimated at `[config] dt = 0.5`, which
/// is not the model's own step, so a replay that found the step in the
/// params file runs at it.
fn fit_toml(method: &str, config: &str) -> String {
    format!(
        r#"store = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[config]
dt = 0.5
{config}

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

{method}
"#
    )
}

const PMMH: &str = r#"[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 1500
burn_in = 300
thin = 1"#;

const IF2: &str = r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 3
cooling = 0.7"#;

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("results"))
        .output()
        .expect("spawn camdl")
}

fn ok(out: &Output, what: &str) -> String {
    assert!(
        out.status.success(),
        "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A project directory holding the model, the data and a fit of `method`
/// labelled `fit`.
fn fitted(tag: &str, method: &str) -> tempfile::TempDir {
    fitted_with(tag, method, "")
}

/// As [`fitted`], with `config` added to the fit file's `[config]`.
fn fitted_with(tag: &str, method: &str, config: &str) -> tempfile::TempDir {
    let tmp = tempfile::Builder::new().prefix(&format!("camdl_point_{tag}_")).tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), fit_toml(method, config)).unwrap();
    ok(&run(tmp.path(), &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]), "fit run");
    tmp
}

/// Each column of the run's `draws.tsv`, by name.
fn draws_columns(store: &Path) -> BTreeMap<String, Vec<f64>> {
    let run = fit_store_support::the_run(store);
    let text = std::fs::read_to_string(run.join("draws.tsv")).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let mut cols: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in lines.filter(|l| !l.is_empty()) {
        for (h, v) in header.iter().zip(line.split('\t')) {
            if *h != "chain" && *h != "draw" {
                cols.entry(h.to_string()).or_default().push(v.parse().unwrap());
            }
        }
    }
    cols
}

/// The top-level values and the `[provenance]` table of a params file.
fn parse(text: &str) -> (BTreeMap<String, f64>, toml::Table) {
    let mut table: toml::Table = text.parse().unwrap();
    let prov = table.remove("provenance").expect("a [provenance] block").as_table().unwrap().clone();
    let values = table.into_iter().map(|(k, v)| (k, v.as_float().or(v.as_integer().map(|i| i as f64)).unwrap())).collect();
    (values, prov)
}

/// The value as the file writes it: ten decimals, trailing zeros dropped.
fn as_written(x: f64) -> f64 {
    format!("{x:.10}").parse().unwrap()
}

#[test]
fn the_posterior_mean_is_the_mean_of_the_draws_and_replays_on_the_fit_simulator() {
    let tmp = fitted("mean", PMMH);
    let dir = tmp.path();
    let text = ok(&run(dir, &["fit", "summary", "@fit", "--point", "posterior-mean"]), "fit summary --point");
    let (values, prov) = parse(&text);

    // Each estimated parameter is the arithmetic mean of its draws; each fixed
    // one is the value the fit held it at.
    let cols = draws_columns(&dir.join("results"));
    for p in ["beta", "gamma"] {
        let xs = &cols[p];
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        assert_eq!(values[p], as_written(mean), "{p}: {text}");
        assert!(xs.iter().any(|x| *x != xs[0]), "{p} must vary across draws for the test to mean anything");
    }
    for (p, v) in [("N0", 10000.0), ("I0", 10.0), ("rho", 0.6), ("k", 10.0)] {
        assert_eq!(values[p], v, "{p}: {text}");
    }
    assert_eq!(values.len(), 6, "every model parameter: {text}");

    // The block names the run, the statistic and the draws, and the fit's
    // simulator.
    let run_dir = fit_store_support::the_run(&dir.join("results"));
    assert_eq!(prov["statistic"].as_str(), Some("posterior_mean"), "{text}");
    assert_eq!(prov["n_draws"].as_integer(), Some(cols["beta"].len() as i64), "{text}");
    assert_eq!(prov["run_id"].as_str(), Some(fit_store_support::run_id(&run_dir).as_str()), "{text}");
    assert_eq!(prov["method"].as_str(), Some("pmmh"), "{text}");
    assert_eq!(prov["backend"].as_str(), Some("chain_binomial"), "{text}");
    assert_eq!(prov["dt"].as_float(), Some(0.5), "{text}");
    assert_eq!(prov["seed"].as_integer(), Some(1), "{text}");
    assert!(!prov.contains_key("chain") && !prov.contains_key("log_likelihood"), "{text}");
    std::fs::write(dir.join("mean.toml"), &text).unwrap();

    // `simulate --params` replays the file on the fit's simulator: with
    // another backend given, its note names the one the block records.
    let out = run(dir, &["simulate", "model.camdl", "--params", "mean.toml", "--backend", "gillespie",
        "--seed", "1", "-o", "t.tsv"]);
    ok(&out, "simulate --params mean.toml");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mean.toml was fitted on chain_binomial at dt = 0.5"), "{stderr}");
    let out = run(dir, &["simulate", "model.camdl", "--params", "mean.toml", "--seed", "1", "-o", "u.tsv"]);
    ok(&out, "simulate --params mean.toml (fit backend)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("backend auto-matched") && stderr.contains("chain_binomial"), "{stderr}");

    // An edited summary is refused, and the advice is to print it again.
    let edited = text.replacen("\nN0 = 10000.0\n", "\nN0 = 10001.0\n", 1);
    assert_ne!(edited, text);
    std::fs::write(dir.join("edited.toml"), &edited).unwrap();
    let out = run(dir, &["simulate", "model.camdl", "--params", "edited.toml", "--seed", "1", "-o", "e.tsv"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    let again = format!("camdl fit summary {} --point posterior-mean", prov["run_id"].as_str().unwrap());
    assert!(stderr.contains(&again) && !stderr.contains("mle_params.toml it writes"), "{stderr}");

    // `predict MODEL --params` predicts at the point, labelled plug_in.
    let out = run(dir, &["predict", "model.camdl", "--params", "mean.toml", "--data", "weekly_cases=weekly_cases.tsv",
        "--mode", "free_forward", "--n-draws", "8"]);
    ok(&out, "predict MODEL --params mean.toml");
    let pred = fit_store_support::predictions(&dir.join("results"))
        .into_iter()
        .map(|d| d.join("predictive").join("weekly_cases.tsv"))
        .find(|p| p.is_file())
        .expect("a predictive table");
    let table = std::fs::read_to_string(&pred).unwrap();
    let header: Vec<&str> = table.lines().next().unwrap().split('\t').collect();
    let col = header.iter().position(|h| *h == "treatment").unwrap();
    let rows: Vec<&str> = table.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r.split('\t').nth(col) == Some("plug_in")), "{table}");
}

#[test]
fn the_posterior_median_is_the_median_of_the_draws() {
    let tmp = fitted("median", PMMH);
    let dir = tmp.path();
    let text = ok(&run(dir, &["fit", "summary", "@fit", "--point", "posterior-median"]), "fit summary --point");
    let (values, prov) = parse(&text);
    let cols = draws_columns(&dir.join("results"));
    for p in ["beta", "gamma"] {
        let mut xs = cols[p].clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        let median = if n % 2 == 1 { xs[n / 2] } else { (xs[n / 2 - 1] + xs[n / 2]) / 2.0 };
        assert!((values[p] - as_written(median)).abs() < 1e-12, "{p}: {} vs {median}: {text}", values[p]);
    }
    assert_eq!(prov["statistic"].as_str(), Some("posterior_median"), "{text}");
}

#[test]
fn an_optimizer_run_is_refused_and_pointed_at_params_only() {
    let tmp = fitted("if2", IF2);
    let out = run(tmp.path(), &["fit", "summary", "@fit", "--point", "posterior-mean"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("optimizer fit") && stderr.contains("camdl fit summary @fit --params-only"), "{stderr}");
    assert!(out.stdout.is_empty(), "nothing printed: {}", String::from_utf8_lossy(&out.stdout));
}

/// A fit made under a setting the block does not record is refused, naming
/// it: `simulate --params` would replay the summary without it.
#[test]
fn a_setting_the_block_cannot_carry_is_refused() {
    let tmp = fitted_with("degenerate", PMMH, "allow_degenerate_rates = true");
    let out = run(tmp.path(), &["fit", "summary", "@fit", "--point", "posterior-mean"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && out.stdout.is_empty(), "{stderr}");
    assert!(stderr.contains("was fitted with allow_degenerate_rates = true")
        && stderr.contains("records only the backend and dt"), "{stderr}");
}

#[test]
fn point_and_params_only_are_refused_together() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run(tmp.path(), &["fit", "summary", "@fit", "--point", "posterior-mean", "--params-only"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("--point") && stderr.contains("--params-only") && stderr.contains("cannot be used with"), "{stderr}");
}
