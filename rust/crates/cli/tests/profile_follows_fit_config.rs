//! `camdl profile --fit` profiles the problem the fit config states (gh#1059).
//!
//! A profile of a fit config is read against `fit run` of the same config, so
//! each cell must score the same data and estimate the same nuisance set:
//!
//! * a `[data] holdout_after` is applied as `fit run` applies it, to the
//!   `--data` binding as to the config's own data, so the held-out rows enter
//!   no cell's likelihood;
//! * `[data.holdout]` files are checked against the bound training data as
//!   `fit run` checks them;
//! * the estimated set is the config's `[estimate]` less the swept and pinned
//!   names, without `--rw-sd`, on any per-cell algorithm; `--rw-sd` is needed
//!   only to name the set when there is no `--fit`;
//! * a `--starts` rule binds the per-cell estimated set, not the swept
//!   parameter, which each cell pins.
//!
//! Every case runs the deterministic `nl-sbplx` cell on the ODE backend
//! except the one that checks IF2 runs without `--rw-sd`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  N0    : count        in [100, 100000]
  I0    : count        in [1, 1000]
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  I = I0
  S = N0 - I0
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

/// Simulated from the model at beta = 0.4, gamma = 0.15, N0 = 10000, I0 = 10,
/// rho = 0.6, k = 10 (chain_binomial, dt = 1, seed 7).
const DATA: &str = "time\tweekly_cases\n7\t47\n14\t111\n21\t474\n28\t1066\n\
                    35\t1471\n42\t533\n49\t300\n56\t40\n";

/// A fit config estimating `beta` and `gamma`; `extra_data` goes into `[data]`.
fn fit_toml(extra_data: &str) -> String {
    format!(
        r#"output_dir = "results"

[model]
camdl = "sir.camdl"

[data]
{extra_data}
[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[method]
algorithm = "nl-sbplx"
backend = "ode"
chains = 1
"#
    )
}

fn workdir(fit: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.path().join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.path().join("fit.toml"), fit).unwrap();
    tmp
}

fn profile(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .current_dir(dir)
        .arg("profile")
        .arg("sir.camdl")
        .args(args)
        .output()
        .expect("spawn camdl profile")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn assert_ok(o: &Output, what: &str) {
    assert!(o.status.success(), "{what}: expected success, got:\n{}", stderr(o));
}

fn assert_refused(o: &Output, what: &str, needles: &[&str]) {
    let err = stderr(o);
    assert!(!o.status.success(), "{what}: must be refused, but exited 0:\n{err}");
    for n in needles {
        assert!(err.contains(n), "{what}: the refusal must mention `{n}`:\n{err}");
    }
}

/// Every file named `name` under `root`, recursively.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == name) {
                found.push(p);
            }
        }
    }
    found
}

const NL: [&str; 4] = ["--algorithm", "nl-sbplx", "--backend", "ode"];

/// `final_loglik` and the `[mle]` table of every cell the run under `dir`
/// wrote, sorted, for comparing two profiles cell for cell.
fn cells(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = find_all(&dir.join("results"), "mle.toml")
        .iter()
        .map(|f| {
            let t: toml::Table = std::fs::read_to_string(f).unwrap().parse().unwrap();
            format!("focal={} final_loglik={} mle={}", t["focal"], t["final_loglik"], t["mle"])
        })
        .collect();
    out.sort();
    out
}

/// Item 1 of gh#1059: with `holdout_after = 35` the fit trains on rows
/// t <= 35, and a profile cell fits exactly those rows, whether the data come
/// from `--data` or from the config's own `[data.observations]`: each cell
/// equals the cell of a profile of the same config without `holdout_after`
/// bound to a file holding only the training rows.
#[test]
fn a_holdout_after_is_applied_as_fit_run_applies_it() {
    let train = "time\tweekly_cases\n7\t47\n14\t111\n21\t474\n28\t1066\n35\t1471\n";
    let base_args = |extra: &[&'static str]| -> Vec<&'static str> {
        let mut v: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.12,0.15",
            "--n-starts", "1", "--starts", "single"];
        v.extend_from_slice(&NL);
        v.extend_from_slice(extra);
        v
    };
    let reference = {
        let tmp = workdir(&fit_toml(""));
        std::fs::write(tmp.path().join("train.tsv"), train).unwrap();
        let o = profile(tmp.path(), &base_args(&["--data", "train.tsv"]));
        assert_ok(&o, "the training rows, no holdout");
        cells(tmp.path())
    };
    assert_eq!(reference.len(), 2, "one cell per grid value: {reference:?}");
    for (what, data) in [
        ("--data on the command line", &["--data", "weekly_cases.tsv"][..]),
        ("data from the fit config", &[][..]),
    ] {
        let tmp = workdir(&fit_toml("holdout_after = 35\n"));
        let o = profile(tmp.path(), &base_args(data));
        assert_ok(&o, what);
        assert_eq!(cells(tmp.path()), reference,
            "{what}: every cell must fit the training rows only");
        assert!(stderr(&o).contains("3 observation row(s) at t > 35 withheld from training"),
            "{what}: the run must say what the holdout withheld:\n{}", stderr(&o));
    }
}

/// `[data.holdout]` files leave the training data whole, so a profile fits what
/// the fit fits; their rows are checked to lie after the training data, as
/// `fit run` checks them. A `--data` file that runs into the held-out rows is
/// refused.
#[test]
fn explicit_holdout_files_are_checked_against_the_bound_data() {
    let tmp = workdir(&fit_toml("[data.holdout]\nweekly_cases = \"holdout.tsv\"\n"));
    let dir = tmp.path();
    std::fs::write(dir.join("train.tsv"),
        "time\tweekly_cases\n7\t47\n14\t111\n21\t474\n28\t1066\n35\t1471\n").unwrap();
    std::fs::write(dir.join("holdout.tsv"),
        "time\tweekly_cases\n42\t533\n49\t300\n56\t40\n").unwrap();
    let mut base: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
        "--rw-sd", "beta=auto", "--n-starts", "1"];
    base.extend_from_slice(&NL);

    let mut overlapping = base.clone();
    overlapping.extend_from_slice(&["--data", "weekly_cases.tsv"]);
    let o = profile(dir, &overlapping);
    assert_refused(&o, "--data overlapping [data.holdout]", &["[data.holdout]", "held-out rows"]);

    let mut training_only = base.clone();
    training_only.extend_from_slice(&["--data", "train.tsv"]);
    let o = profile(dir, &training_only);
    assert_ok(&o, "--data ending before [data.holdout]");
    assert!(stderr(&o).contains("held-out observation row(s) bound as the tail"),
        "the run must say the holdout was checked:\n{}", stderr(&o));
}

/// Items 3 and 4 of gh#1059: `gamma` is listed in `[estimate]` and swept, so
/// each cell pins it and estimates `beta` alone. A `from_posterior` source
/// with a `beta` column and no `gamma` column binds every parameter a cell
/// estimates and must be accepted; every chain then starts at its row.
#[test]
fn a_swept_estimate_parameter_needs_no_column_in_a_starts_source() {
    let tmp = workdir(&fit_toml(""));
    let dir = tmp.path();
    std::fs::write(dir.join("draws.tsv"), "chain\tbeta\n0\t0.33\n1\t0.33\n").unwrap();
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.12,0.15",
        "--rw-sd", "beta=auto", "--n-starts", "2", "--starts", "from_posterior=draws.tsv"];
    args.extend_from_slice(&NL);
    let o = profile(dir, &args);
    assert_ok(&o, "from_posterior without the swept column");
    let runs = find_all(&dir.join("results"), "run.json");
    assert!(!runs.is_empty(), "no run.json written");
    for run in &runs {
        let text = std::fs::read_to_string(run).unwrap();
        assert!(text.contains("0.33"),
            "every chain must start at the source's beta = 0.33 ({}):\n{text}", run.display());
    }
}

/// Listing the swept parameter in `[fixed]` stays refused, naming the block.
#[test]
fn a_swept_fixed_parameter_is_refused() {
    let fit = fit_toml("").replace("rho = 0.6\n", "rho = 0.6\ngamma = 0.15\n")
        .replace("gamma = { bounds = [0.01, 0.5], start = 0.15 }\n", "");
    let tmp = workdir(&fit);
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.12,0.15",
        "--rw-sd", "beta=auto", "--n-starts", "1"];
    args.extend_from_slice(&NL);
    let o = profile(tmp.path(), &args);
    assert_refused(&o, "a swept [fixed] parameter", &["'gamma' is in both `--sweep`", "[fixed]",
        "a swept parameter is listed in [estimate]"]);
}
/// The `[mle]` table of every `mle.toml` the run wrote.
fn mle_tables(dir: &Path) -> Vec<toml::Table> {
    let files = find_all(&dir.join("results"), "mle.toml");
    assert!(!files.is_empty(), "no mle.toml under {}", dir.display());
    files
        .iter()
        .map(|f| {
            let t: toml::Table = std::fs::read_to_string(f).unwrap().parse().unwrap();
            t["mle"].as_table().expect("[mle]").clone()
        })
        .collect()
}

/// Item 2 of gh#1059: an nl-* cell reads no step, so with `--fit` it needs no
/// `--rw-sd`; it estimates `[estimate]` less the swept `gamma`, i.e. `beta`.
#[test]
fn an_nlopt_profile_of_a_fit_config_needs_no_rw_sd() {
    let tmp = workdir(&fit_toml(""));
    let dir = tmp.path();
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.12,0.15",
        "--n-starts", "1", "--starts", "single"];
    args.extend_from_slice(&NL);
    let o = profile(dir, &args);
    assert_ok(&o, "nl-sbplx --fit without --rw-sd");
    for mle in mle_tables(dir) {
        let names: Vec<&String> = mle.keys().collect();
        assert_eq!(names, vec!["beta"], "each cell estimates [estimate] less the swept gamma");
    }
}

/// An IF2 profile of a fit config steps each parameter by its `[estimate]`
/// `rw_sd` or the automatic default, as `fit run` does; `--rw-sd` is optional.
#[test]
fn an_if2_profile_of_a_fit_config_takes_the_automatic_step() {
    let tmp = workdir(&fit_toml(""));
    let dir = tmp.path();
    let o = profile(dir, &["--fit", "fit.toml", "--sweep", "gamma=0.15", "--n-starts", "1",
        "--starts", "single", "--particles", "50", "--iterations", "2"]);
    assert_ok(&o, "if2 --fit without --rw-sd");
    let tables = mle_tables(dir);
    assert_eq!(tables.len(), 1);
    assert!(tables[0].contains_key("beta") && !tables[0].contains_key("gamma"),
        "the cell estimates beta and pins gamma: {:?}", tables[0]);
}

/// Without `--fit` nothing but `--rw-sd` names the estimated set.
#[test]
fn without_a_fit_config_the_estimated_set_must_be_named() {
    let tmp = workdir(&fit_toml(""));
    let mut args: Vec<&str> = vec!["--data", "weekly_cases.tsv", "--sweep", "gamma=0.15",
        "--fixed", "N0=10000", "--fixed", "I0=10", "--fixed", "rho=0.6", "--fixed", "k=10"];
    args.extend_from_slice(&NL);
    let o = profile(tmp.path(), &args);
    assert_refused(&o, "no --fit, no --rw-sd", &["--fit", "--rw-sd NAME=auto"]);
}

/// With `--fit`, an explicit `--rw-sd` list that leaves out an `[estimate]`
/// parameter would hold it at its start in every cell, while the config says
/// it is estimated: refused, naming it.
#[test]
fn an_rw_sd_list_short_of_the_fit_estimate_is_refused() {
    let fit = fit_toml("").replace(
        "[fixed]\nN0 = 10000\nI0 = 10\nrho = 0.6\n",
        "rho = { bounds = [0.1, 0.9], start = 0.6 }\n\n[fixed]\nN0 = 10000\nI0 = 10\n");
    let tmp = workdir(&fit);
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
        "--rw-sd", "beta=auto", "--n-starts", "1", "--starts", "single"];
    args.extend_from_slice(&NL);
    let o = profile(tmp.path(), &args);
    assert_refused(&o, "--rw-sd beta with rho also in [estimate]", &["leaves out rho"]);
}

/// Item 6 of gh#1059: `fit run` takes no `--fixed` (a held parameter is a
/// `[fixed]` entry of the config), and `profile --help` must not say it does.
#[test]
fn profile_help_does_not_offer_fixed_on_fit_run() {
    let tmp = workdir(&fit_toml(""));
    let o = Command::new(binary())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .current_dir(tmp.path())
        .args(["fit", "run", "fit.toml", "--fixed", "gamma=0.15"])
        .output()
        .unwrap();
    assert!(!o.status.success(), "fit run --fixed must be rejected");
    let help = profile(tmp.path(), &["--help"]);
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("`camdl fit run` takes no `--fixed`"),
        "profile --help must say fit run takes no --fixed:\n{text}");
    assert!(!text.contains("(if2, profile, fit run)"),
        "profile --help still lists fit run for --fixed:\n{text}");
}

/// An explicit `--rw-sd` list is a hash map; the cells must estimate its
/// names in one order (the model's), or a start rule that draws per parameter
/// in that order hands each run's draws to different parameters, and the same
/// invocation, with the same run key, stores different estimates from one
/// process to the next. Four runs in fresh directories must agree byte for
/// byte.
#[test]
fn an_explicit_rw_sd_list_profiles_reproducibly() {
    let fit = fit_toml("").replace(
        "[fixed]\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n",
        "rho = { bounds = [0.1, 0.9], start = 0.6 }\nk = { bounds = [1.0, 100.0], start = 10.0 }\n\n\
         [fixed]\nN0 = 10000\nI0 = 10\n");
    let mut seen: Vec<Vec<String>> = Vec::new();
    for _ in 0..4 {
        let tmp = workdir(&fit);
        let dir = tmp.path();
        let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
            "--rw-sd", "beta=auto,rho=auto,k=auto", "--n-starts", "2", "--starts", "uniform"];
        args.extend_from_slice(&NL);
        let o = profile(dir, &args);
        assert_ok(&o, "explicit --rw-sd list");
        let mut files = find_all(&dir.join("results"), "mle.toml");
        files.sort();
        seen.push(files.iter().map(|f| std::fs::read_to_string(f).unwrap()).collect());
    }
    for run in &seen[1..] {
        assert_eq!(run, &seen[0], "the same invocation must store the same cells");
    }
}

/// The starts rule every cell's chains were drawn under, from each run.json's
/// `init_provenance`.
fn starts_rules(dir: &Path) -> Vec<String> {
    fn find(v: &serde_json::Value) -> Option<String> {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(ip) = m.get("init_provenance") {
                    return ip.get("method").and_then(|s| s.as_str()).map(String::from);
                }
                m.values().find_map(find)
            }
            serde_json::Value::Array(a) => a.iter().find_map(find),
            _ => None,
        }
    }
    let runs = find_all(&dir.join("results"), "run.json");
    assert!(!runs.is_empty(), "no run.json under {}", dir.display());
    runs.iter()
        .map(|r| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(r).unwrap()).unwrap();
            find(&v).unwrap_or_else(|| panic!("no init_provenance in {}", r.display()))
        })
        .collect()
}

/// gh#1059: without `--starts` a profile draws each cell's chains by the rule
/// `fit run` defaults to: `from_prior` when every parameter a cell estimates
/// has a prior to draw from, `uniform_unconstrained` otherwise. The swept
/// parameter is pinned, so its prior (or lack of one) does not count.
#[test]
fn the_default_starts_rule_is_the_one_fit_run_defaults_to() {
    let with_prior = fit_toml("").replace(
        "beta = { bounds = [0.05, 1.0], start = 0.4 }",
        "beta = { bounds = [0.05, 1.0], start = 0.4, prior = { uniform = {} } }");
    for (fit, rule, line) in [
        (with_prior, "from_prior", "every estimated parameter declares a prior"),
        (fit_toml(""), "uniform_unconstrained", "no sampleable prior on beta"),
    ] {
        let tmp = workdir(&fit);
        let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.12,0.15",
            "--n-starts", "2"];
        args.extend_from_slice(&NL);
        let o = profile(tmp.path(), &args);
        assert_ok(&o, rule);
        assert!(starts_rules(tmp.path()).iter().all(|r| r == rule),
            "every cell must start under `{rule}`: {:?}", starts_rules(tmp.path()));
        assert!(stderr(&o).contains(line), "the run must say why `{rule}`:\n{}", stderr(&o));
    }
}

/// A draws file of three chains: 0 and 1 agree on `beta` near 0.33, chain 2
/// sits at 0.9.
const THREE_CHAINS: &str = "chain\tbeta\n\
    0\t0.330\n0\t0.332\n0\t0.331\n0\t0.333\n0\t0.330\n0\t0.332\n\
    1\t0.331\n1\t0.333\n1\t0.330\n1\t0.332\n1\t0.331\n1\t0.333\n\
    2\t0.900\n2\t0.902\n2\t0.901\n2\t0.903\n2\t0.900\n2\t0.902\n";

/// Every `beta` start value recorded in the run.json files under `dir`.
fn beta_starts(dir: &Path) -> Vec<f64> {
    fn walk(v: &serde_json::Value, out: &mut Vec<f64>) {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(chains) = m.get("init_provenance").and_then(|ip| ip.get("chains")) {
                    for c in chains.as_array().into_iter().flatten() {
                        if let Some(b) = c.get("beta").and_then(|b| b.get("value")) {
                            out.push(b.as_f64().unwrap());
                        }
                    }
                    return;
                }
                m.values().for_each(|x| walk(x, out));
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for r in find_all(&dir.join("results"), "run.json") {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(r).unwrap()).unwrap();
        walk(&v, &mut out);
    }
    out
}

/// gh#1059 (gh#1056 on `profile`): `--starts-chains` restricts a
/// `from_posterior` rule to those chains of its source, as on `fit run`, and
/// the subset is keyed: the same profile without it is a different run.
#[test]
fn starts_chains_restricts_a_from_posterior_rule() {
    let tmp = workdir(&fit_toml(""));
    let dir = tmp.path();
    std::fs::write(dir.join("draws.tsv"), THREE_CHAINS).unwrap();
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
        "--n-starts", "6", "--starts", "from_posterior=draws.tsv", "--starts-chains", "0,1"];
    args.extend_from_slice(&NL);
    let o = profile(dir, &args);
    assert_ok(&o, "--starts-chains 0,1");
    let starts = beta_starts(dir);
    assert!(!starts.is_empty(), "no recorded starts");
    assert!(starts.iter().all(|b| (0.329..0.334).contains(b)),
        "every start must come from chains 0 and 1: {starts:?}");

    // Without the subset, the same profile is another run: its leaves land
    // beside the first ones rather than on them.
    let before = find_all(&dir.join("results"), "run.json").len();
    let whole: Vec<&str> = args.iter().copied()
        .filter(|a| *a != "--starts-chains" && *a != "0,1").collect();
    assert_ok(&profile(dir, &whole), "the whole cloud");
    assert_eq!(find_all(&dir.join("results"), "run.json").len(), 2 * before,
        "the chain subset must be part of the run's identity");
}

/// A chain subset names chains of a `from_posterior` source; on any other
/// rule, the default included, it is refused, as on `fit run`.
#[test]
fn starts_chains_without_from_posterior_is_refused() {
    let tmp = workdir(&fit_toml(""));
    for extra in [&[][..], &["--starts", "uniform"][..]] {
        let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
            "--n-starts", "2", "--starts-chains", "0,1"];
        args.extend_from_slice(&NL);
        args.extend_from_slice(extra);
        let o = profile(tmp.path(), &args);
        assert_refused(&o, "--starts-chains without from_posterior",
            &["--starts-chains", "from_posterior"]);
    }
}

/// gh#1059: a starts source that is not converged is refused unless
/// `--allow-nonconverged-source` is passed, as on `fit run`. Chains 0 and 2
/// of the source disagree on `beta`, so their own R̂ fails the gate.
#[test]
fn a_nonconverged_starts_source_is_refused_unless_allowed() {
    let tmp = workdir(&fit_toml(""));
    let dir = tmp.path();
    std::fs::write(dir.join("draws.tsv"), THREE_CHAINS).unwrap();
    let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
        "--n-starts", "2", "--starts", "from_posterior=draws.tsv", "--starts-chains", "0,2"];
    args.extend_from_slice(&NL);
    let o = profile(dir, &args);
    assert_refused(&o, "a disagreeing chain subset",
        &["the selected chains disagree", "--allow-nonconverged-source"]);
    assert!(!stderr(&o).contains("camdl fit run"),
        "the remedy must not name another command:\n{}", stderr(&o));
    assert!(find_all(&dir.join("results"), "mle.toml").is_empty(),
        "a refused profile must store no cell");

    args.push("--allow-nonconverged-source");
    let o = profile(dir, &args);
    assert_ok(&o, "--allow-nonconverged-source");
    assert!(stderr(&o).contains("starting from them anyway"),
        "the run must say it started from them anyway:\n{}", stderr(&o));
}

/// The flag lifts the refusal of a sourced rule's verdict; on a rule with no
/// verdict, the default included, it would change nothing and is refused.
#[test]
fn allow_nonconverged_source_without_a_fit_source_is_refused() {
    let tmp = workdir(&fit_toml(""));
    for extra in [&[][..], &["--starts", "uniform"][..]] {
        let mut args: Vec<&str> = vec!["--fit", "fit.toml", "--sweep", "gamma=0.15",
            "--n-starts", "2", "--allow-nonconverged-source"];
        args.extend_from_slice(&NL);
        args.extend_from_slice(extra);
        let o = profile(tmp.path(), &args);
        assert_refused(&o, "--allow-nonconverged-source without a source",
            &["--allow-nonconverged-source", "would change nothing"]);
    }
}
