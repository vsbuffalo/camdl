//! `baseline` is an ordinary scenario name on every verb that takes
//! `--scenario`.
//!
//! The book's models declare `scenarios { baseline { set = { ... } } }`, so
//! `--scenario baseline` has to mean that scenario, and on a model that does
//! not declare one it has to be refused like any other undeclared name, with
//! the declared scenarios listed. The no-scenario arm is the parenthesised
//! `(none)` (`simulate`) or the compiler-reserved `fitted` (`fit predict`),
//! neither of which a declared scenario can be called.
//!
//! The observable is the initial infected count: the params set `I0 = 5`, the
//! model's `baseline` scenario sets `I0 = 3`, and a quantity reads `I` at
//! time 0. A 3 means the scenario was applied, a 5 that it was ignored.
//!
//! Shells out to the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Where the latest `fit predict` of a run in `seg` wrote:
/// `<seg>/predict/<run id>[-<options key>]/` (gh#896), the directory whose
/// `report*.json` (one written by every predict) is newest. A path that does not
/// exist while nothing has been predicted.
fn predict_out(seg: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(seg.join("predict"))
        .into_iter().flatten().flatten().map(|e| e.path())
        .filter_map(|d| {
            // Newest `report*.json` (keyed by chain selection) in the directory.
            let t = std::fs::read_dir(&d).ok()?.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.starts_with("report") && n.ends_with(".json")
                })
                .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
                .max()?;
            Some((t, d))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, d)| d)
        .unwrap_or_else(|| seg.join("predict").join("none"))
}


fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

/// The model, with its one scenario named `scenario_name`.
fn model(scenario_name: &str) -> String {
    format!(
        r#"
time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}}

let N = S + I + R

transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}

init {{
  S = N0 - I0
  I = I0
}}

observations {{
  weekly_cases {{
    columns       {{ time : time, weekly_cases : count }}
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }}
}}

quantities {{
  I_start = value_at(I, 0)
}}

scenarios {{
  {scenario_name} {{ set = {{ I0 = 3 }} }}
}}

simulate {{
  from = 0 'days
  to   = 56 'days
}}
"#
    )
}

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const PARAMS: &str = "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 5\nrho = 0.6\nk = 10\n";

const FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 5
rho = 0.6
k   = 10.0

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 50
sweeps = 20
burn_in = 5
thin = 1
"#;

fn setup(scenario_name: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.camdl"), model(scenario_name)).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("params.toml"), PARAMS).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), FIT_TOML).unwrap();
    tmp
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        // Ad-hoc run: skip the camdlc git-hash handshake (the binary under test
        // is self-consistent).
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn fit(bin: &Path, dir: &Path) {
    ok(&run(bin, dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
}

/// The single `quantities/I_start.tsv` under the fit's results, as
/// scenario → q50.
fn i_start_by_scenario(dir: &Path) -> std::collections::BTreeMap<String, String> {
    let fits = dir.join("results").join("fits");
    let path = std::fs::read_dir(&fits)
        .unwrap()
        .flatten()
        .map(|e| predict_out(&e.path()).join("quantities").join("I_start.tsv"))
        .find(|p| p.is_file())
        .expect("quantities/I_start.tsv must be written");
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|c| *c == name).unwrap();
    let (scen, q50) = (col("scenario"), col("q50"));
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[scen].to_string(), f[q50].to_string())
        })
        .collect()
}

#[test]
fn fit_predict_applies_a_declared_baseline_scenario() {
    let bin = skip_if_missing_binary();
    let tmp = setup("baseline");
    let dir = tmp.path();
    fit(&bin, dir);
    let out = run(&bin, dir, &[
        "fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward",
        "--scenario", "baseline", "--seed", "1",
    ]);
    ok(&out, "fit predict --scenario baseline");
    let by = i_start_by_scenario(dir);
    assert_eq!(by.get("baseline").map(String::as_str), Some("3"),
        "the declared `baseline` scenario sets I0 = 3: {by:?}");
    assert_eq!(by.get("fitted").map(String::as_str), Some("5"),
        "the no-overlay `fitted` arm keeps the fit's I0 = 5: {by:?}");
}

#[test]
fn simulate_applies_a_declared_baseline_scenario() {
    let bin = skip_if_missing_binary();
    let tmp = setup("baseline");
    let dir = tmp.path();
    let out = run(&bin, dir, &[
        "simulate", "model.camdl", "--params", "params.toml", "--scenario", "baseline",
        "--backend", "ode", "--seed", "1", "-o", "traj.tsv",
    ]);
    ok(&out, "simulate --scenario baseline");
    let txt = std::fs::read_to_string(dir.join("traj.tsv")).unwrap();
    let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let i = header.iter().position(|c| *c == "I").unwrap();
    let first: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(first[i], "3", "the declared `baseline` scenario sets I0 = 3");
}

/// Assert `out` is a refusal of the undeclared `baseline` that names the
/// declared scenario `other`.
fn refused(out: &Output, verb: &str) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!out.status.success(), "{verb} must refuse an undeclared --scenario baseline:\n{stderr}");
    assert!(stderr.contains("'baseline'"), "{verb}: names the refused scenario:\n{stderr}");
    assert!(stderr.contains("other"), "{verb}: lists the declared scenarios:\n{stderr}");
    stderr
}

#[test]
fn an_undeclared_baseline_is_refused_by_every_verb() {
    let bin = skip_if_missing_binary();
    let tmp = setup("other");
    let dir = tmp.path();

    refused(&run(&bin, dir, &[
        "simulate", "model.camdl", "--params", "params.toml", "--scenario", "baseline",
        "--backend", "ode", "--seed", "1", "-o", "traj.tsv",
    ]), "simulate");
    assert!(!dir.join("traj.tsv").exists(), "simulate wrote nothing");

    // `--scenario` supplies the parameters in pfilter, so it conflicts with
    // `--params`; the refusal comes before any parameter is needed.
    refused(&run(&bin, dir, &[
        "pfilter", "model.camdl", "--data", "weekly_cases.tsv", "--particles", "20",
        "--scenario", "baseline", "--seed", "1",
    ]), "pfilter");

    refused(&run(&bin, dir, &[
        "profile", "model.camdl", "--starts", "from_params=params.toml",
        "--data", "weekly_cases.tsv", "--particles", "20", "--rw-sd", "auto",
        "--sweep", "beta=lin(0.3,0.4,2)", "--scenario", "baseline", "--seed", "1",
    ]), "profile");

    refused(&run(&bin, dir, &[
        "survey", "model.camdl", "--parallel", "1", "--data", "weekly_cases.tsv",
        "--estimate", "beta=0.1:0.9", "--estimate", "gamma=0.05:0.3",
        "--fixed", "N0=10000", "--fixed", "I0=5", "--fixed", "rho=0.6", "--fixed", "k=10",
        "--n-points", "10", "--scenario", "baseline",
    ]), "survey");

    // fit predict refuses up front, before any replay runs or any file is
    // written — not partway through, with partial output on disk.
    fit(&bin, dir);
    let stderr = refused(&run(&bin, dir, &[
        "fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward",
        "--scenario", "baseline", "--seed", "1",
    ]), "fit predict");
    assert!(stderr.contains("is not declared in the model"),
        "fit predict: the up-front scenario check refuses it:\n{stderr}");
    assert!(!stderr.contains("wrote "), "fit predict wrote nothing:\n{stderr}");
}
