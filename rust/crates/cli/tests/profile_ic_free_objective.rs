//! `profile --fit` and `survey --fit` score the likelihood the fit config
//! defines, including `ic_free`.
//!
//! `ic_free = true` makes `fit run` maximise the conditional likelihood
//! `log p(y₂:T | y₁, θ)`: the filter weights and resamples at y₁ but leaves
//! y₁'s term out. A profile of that fit must score the same objective, or its
//! curve sits `log p(y₁ | θ)` away from the fit's maximum — several nats here —
//! and the two cannot be compared.
//!
//! The check is a known value. At the fit's own MLE θ̂, with every parameter
//! pinned so the profile cell only evaluates θ̂:
//!
//! * `pfilter` on the full series gives `log p(y₁:T | θ̂)`;
//! * `pfilter` on the first row alone gives `log p(y₁ | θ̂)`;
//! * their difference is the conditional likelihood, which is what `fit run`
//!   reports and what the profile cell must store.
//!
//! The particle-filter estimates differ by Monte Carlo noise (a 500-particle
//! clean evaluation in the profile cell); the tolerance is well inside the
//! first observation's term, which the test also requires to be large.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

/// A closed SIR whose initial infecteds are drawn (`I ~ poisson`), so the
/// swarm has the spread at t=0 that `ic_free` conditions on.
const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
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
  I ~ poisson(rate = I0)
  S = N0 - I
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

const FIT: &str = r#"ic_free = true
output_dir = "results"

[model]
camdl = "sir.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0 = 10000
I0 = 10
rho = 0.6
k = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 300
iterations = 10
cooling = 0.5
"#;

fn camdl(bin: &Path, dir: &Path) -> Command {
    let mut c = Command::new(bin);
    c.env("CAMDL_SKIP_VERSION_CHECK", "1");
    c.env("CAMDLC", camdlc_bin());
    c.current_dir(dir);
    c
}

fn run(cmd: &mut Command, what: &str) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("spawn {what}: {e}"));
    assert!(
        out.status.success(),
        "{what} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Every file named `name` under `root`, recursively.
fn find_all(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
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

/// `camdl pfilter` at the MLE over `data`: the log-likelihood it prints.
fn pfilter_loglik(bin: &Path, dir: &Path, mle: &Path, data: &str) -> f64 {
    let stdout = run(
        camdl(bin, dir).args([
            "pfilter", "sir.camdl",
            "--params", mle.to_str().unwrap(),
            "--data", data,
            "--particles", "20000",
            "--seed", "11",
        ]),
        "pfilter",
    );
    stdout
        .lines()
        .find_map(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("pfilter printed no log-likelihood:\n{stdout}"))
}

#[test]
fn profile_scores_the_ic_free_objective_of_the_fit_it_profiles() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let dir = tempfile::tempdir().expect("tempdir");
    let d = dir.path();
    std::fs::write(d.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(d.join("weekly_cases.tsv"), DATA).unwrap();
    let first_row: String = DATA.lines().take(2).map(|l| format!("{}\n", l.trim())).collect();
    std::fs::write(d.join("first_obs.tsv"), first_row).unwrap();
    std::fs::write(d.join("fit.toml"), FIT).unwrap();

    run(camdl(&bin, d).args(["fit", "run", "fit.toml", "--seed", "3", "--no-progress"]), "fit run");
    let mle_paths = find_all(&d.join("results/fits"), "mle_params.toml");
    assert_eq!(mle_paths.len(), 1, "one fit leaf expected: {mle_paths:?}");
    let mle_path = &mle_paths[0];
    let mle: toml::Value = toml::from_str(&std::fs::read_to_string(mle_path).unwrap()).unwrap();
    let beta = mle["beta"].as_float().unwrap();
    let gamma = mle["gamma"].as_float().unwrap();
    let fit_ll = mle["provenance"]["log_likelihood"].as_float().unwrap();

    // The known value: the conditional likelihood at θ̂, as `pfilter` computes
    // it from the full series and the first row alone.
    let full = pfilter_loglik(&bin, d, mle_path, "weekly_cases.tsv");
    let first = pfilter_loglik(&bin, d, mle_path, "first_obs.tsv");
    let conditional = full - first;
    assert!(
        first < -2.0,
        "the first observation's term must be large enough to detect: log p(y₁|θ̂) = {first}"
    );
    assert!(
        (fit_ll - conditional).abs() < 0.5,
        "fit run must report the conditional likelihood: fit {fit_ll}, \
         pfilter full {full} − first {first} = {conditional}"
    );

    // The profile at θ̂: focal beta at the fit's value, gamma pinned at the
    // fit's value, so the one cell scores exactly θ̂.
    run(
        camdl(&bin, d).args([
            "profile", "sir.camdl",
            "--fit", "fit.toml",
            "--sweep", &format!("beta={beta},{}", beta + 0.05),
            "--fixed", &format!("gamma={gamma}"),
            "--particles", "500",
            "--iterations", "2",
            "--n-starts", "1",
            "--rw-sd", "auto",
            "--no-progress",
        ]),
        "profile",
    );
    let cells: Vec<serde_json::Value> = find_all(&d.join("results/profiles"), "run.json")
        .iter()
        .map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap())
        .collect();
    let at_mle = cells
        .iter()
        .find(|c| c["inputs"]["grid_point"] == 0)
        .expect("a profile cell at the fit's beta");
    let profile_ll = at_mle["inputs"]["best_loglik"].as_f64().expect("finite cell loglik");

    assert_eq!(
        at_mle["inputs"]["likelihood_objective"], "conditional_on_first_observation",
        "the cell record must name the objective it scored"
    );
    assert!(
        (profile_ll - conditional).abs() < 1.0 && (profile_ll - fit_ll).abs() < 1.0,
        "the profile at the fit's MLE must score the fit's conditional likelihood: \
         profile {profile_ll}, fit {fit_ll}, pfilter conditional {conditional}, \
         pfilter full {full}"
    );
}

#[test]
fn survey_of_an_ic_free_fit_is_a_different_landscape() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let dir = tempfile::tempdir().expect("tempdir");
    let d = dir.path();
    std::fs::write(d.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(d.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(d.join("conditional.toml"), FIT).unwrap();
    std::fs::write(d.join("full.toml"), FIT.replace("ic_free = true", "ic_free = false")).unwrap();

    // Same problem, same points, same seed — the same particle paths, since
    // leaving a term out of the sum draws no random numbers. Point by point
    // the landscapes therefore differ by exactly that point's estimate of
    // `log p(y₁ | θ)`, which is negative for a count observation.
    let landscape = |fit: &str| -> (serde_json::Value, std::collections::BTreeMap<u64, f64>) {
        run(
            camdl(&bin, d).args([
                "survey", "sir.camdl", "--fit", fit,
                "--n-points", "8", "--eval", "pfilter",
                "--eval-particles", "300", "--eval-replicates", "1",
                "--no-progress", "--output", &format!("out-{fit}"),
            ]),
            "survey",
        );
        let runs = find_all(&d.join(format!("out-{fit}")), "run.json");
        assert_eq!(runs.len(), 1, "one survey leaf expected: {runs:?}");
        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&runs[0]).unwrap()).unwrap();
        let tsv = std::fs::read_to_string(runs[0].with_file_name("landscape.tsv")).unwrap();
        let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let ll = header.iter().position(|h| *h == "loglik").expect("loglik column");
        let id = header.iter().position(|h| *h == "point_id").expect("point_id column");
        let rows = lines
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                (f[id].parse::<u64>().unwrap(), f[ll].parse::<f64>().unwrap_or(f64::NEG_INFINITY))
            })
            .collect();
        (rec, rows)
    };
    let (cond_rec, cond) = landscape("conditional.toml");
    let (full_rec, full) = landscape("full.toml");

    assert_eq!(cond_rec["inputs"]["likelihood_objective"], "conditional_on_first_observation");
    assert_eq!(full_rec["inputs"]["likelihood_objective"], "full");
    let scored: Vec<(f64, f64)> = cond
        .iter()
        .filter_map(|(id, c)| full.get(id).map(|f| (*c, *f)))
        .filter(|(c, f)| c.is_finite() && f.is_finite())
        .collect();
    assert!(!scored.is_empty(), "no point scored under both objectives");
    for (c, f) in &scored {
        assert!(
            c > f,
            "leaving y₁'s term out must raise every point's loglik: conditional {c}, full {f}"
        );
    }
}
