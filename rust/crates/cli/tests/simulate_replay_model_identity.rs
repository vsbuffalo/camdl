//! `simulate` replays a fit's parameters through the model named on the
//! command line. Those parameters were estimated under one model — the one
//! the fit archived as `<segment>/model.ir.json`, or, for a fit config, the
//! one its `[model]` names — so a different model must be refused before
//! anything runs. Before this check, `simulate edited.camdl --draws posterior
//! --fit <fit>` with the recovery rate tripled replayed the posterior through
//! the edited dynamics at exit 0 (peak I 10–258 per draw against 1296–3673
//! through the fitted model).
//!
//! The comparison is the one `fit predict --quantities` uses
//! (`fit::fitted_model::compare_to_fitted`): quantities and contrasts are
//! excluded, so a reporting-only edit runs and says so.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

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
I0  = 10
rho = 0.6
k   = 10.0
[method]
algorithm = "mh"
backend = "ode"
chains = 1
iterations = 60
burn_in = 20
starts = "single"
"#;

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// One fit, shared by every case: the posterior, the archive and the config
/// are what each replay is checked against.
struct Fixture {
    dir: PathBuf,
    segment: String,
}

fn fixture(tag: &str) -> Fixture {
    fixture_of(tag, MODEL)
}

fn fixture_of(tag: &str, model: &str) -> Fixture {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let dir = std::env::temp_dir().join(format!("camdl_replay_model_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    // The recovery rate tripled: different dynamics, same parameters.
    std::fs::write(
        dir.join("edited.camdl"),
        model.replace("@ gamma * I", "@ 3 * gamma * I"),
    )
    .unwrap();
    // A reporting-only edit: a quantities block, nothing else.
    std::fs::write(dir.join("quant.camdl"), format!("{model}\nquantities {{\n  peak_I = max(I)\n}}\n"))
        .unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();
    std::fs::write(dir.join("d.tsv"), "beta\tgamma\n0.3\t0.1\n0.5\t0.2\n").unwrap();
    let out = run(&dir, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let fits = dir.join("results").join("fits");
    let segment = std::fs::read_dir(&fits)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("fit segment dir");
    let segment = segment.strip_prefix(&dir).unwrap().to_string_lossy().into_owned();
    Fixture { dir, segment }
}

fn simulate(f: &Fixture, model: &str, rest: &[&str]) -> (bool, String) {
    let mut args = vec!["simulate", model];
    args.extend_from_slice(rest);
    args.extend_from_slice(&["--seed", "3", "--output-dir", "out"]);
    let out = run(&f.dir, &args);
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn a_replay_through_another_model_is_refused_and_the_fitted_one_runs() {
    let f = fixture("all");
    let seg = f.segment.as_str();
    let archive = format!("{seg}/model.ir.json");

    // `--draws posterior --fit <stored fit>` through the edited model: refused
    // before running, naming the fit, both identities and the archived IR.
    let (ok, err) = simulate(&f, "edited.camdl", &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
    assert!(!ok, "a posterior replay through a different model must be refused:\n{err}");
    assert!(err.contains("edited.camdl is not the model this fit ran on"), "{err}");
    assert!(err.contains(&format!("--fit {seg}")), "names the fit:\n{err}");
    assert!(err.contains(&archive), "names the archived model as the fix:\n{err}");
    assert!(err.contains("fitted model ") && err.contains("edited.camdl "), "both identities:\n{err}");
    assert!(!f.dir.join("out").join("sims").exists(), "nothing ran:\n{err}");

    // The same through a fit.toml handle naming the stored run.
    let (ok, err) =
        simulate(&f, "edited.camdl", &["--draws", "posterior", "--fit", "fit.toml", "--n-draws", "2"]);
    assert!(!ok && err.contains("is not the model this fit ran on"), "{err}");

    // The fitted model runs: as source, and as the archived IR the refusal names.
    for model in ["model.camdl", archive.as_str()] {
        let (ok, err) = simulate(&f, model, &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
        assert!(ok, "{model} is the fitted model and must replay:\n{err}");
        assert!(!err.contains("note: "), "no notice for the same model:\n{err}");
    }

    // `--output-every` rewrites this run's output schedule, not the model: the
    // check compares the model as named.
    let (ok, err) = simulate(
        &f,
        "model.camdl",
        &["--draws", "posterior", "--fit", seg, "--n-draws", "2", "--output-every", "2"],
    );
    assert!(ok, "--output-every must not read as a different model:\n{err}");

    // A quantities-only difference runs, and says whose vocabulary applies.
    let (ok, err) = simulate(&f, "quant.camdl", &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
    assert!(ok, "a reporting-only edit must run:\n{err}");
    assert!(err.contains("quant.camdl's quantities and contrasts apply"), "{err}");

    // `--draws <file> --fit <stored fit>`: the fit's [fixed] block is replayed
    // through the command-line model — the same check.
    let (ok, err) = simulate(&f, "edited.camdl", &["--draws", "d.tsv", "--fit", seg]);
    assert!(!ok && err.contains("is not the model this fit ran on"), "{err}");
    let (ok, err) = simulate(&f, "model.camdl", &["--draws", "d.tsv", "--fit", seg]);
    assert!(ok, "{err}");

    // `--draws <file>|prior --fit <fit.toml>`: the config's [fixed] and priors
    // are written for the model its [model] names.
    for draws in [&["--draws", "d.tsv"][..], &["--draws", "prior", "-n", "2"][..]] {
        let mut rest = draws.to_vec();
        rest.extend_from_slice(&["--fit", "fit.toml"]);
        let (ok, err) = simulate(&f, "edited.camdl", &rest);
        assert!(!ok, "{draws:?}: a config beside another model must be refused:\n{err}");
        assert!(err.contains("edited.camdl is not the model this fit config names"), "{err}");
        assert!(err.contains("Fix: run with model.camdl"), "{err}");
        let (ok, err) = simulate(&f, "model.camdl", &rest);
        assert!(ok, "{draws:?}: the config's own model must run:\n{err}");
    }

    let _ = std::fs::remove_dir_all(&f.dir);
}

/// An observation-anchored model (`simulate { to = last_obs + 2 'weeks }`).
/// `fit run` resolves the anchor against its `[data]` and archives the
/// SUBSTITUTED model — horizon 70, the marker cleared — so the command-line
/// source, which still carries the marker, has to go through the same
/// substitution before it can be compared. Without it the fit's own,
/// untouched source was refused as "edited since".
#[test]
fn an_anchored_fit_replays_through_its_own_source() {
    let anchored = MODEL.replace("to = 80 'days", "to = last_obs + 2 'weeks");
    assert_ne!(anchored, MODEL);
    let f = fixture_of("anchored", &anchored);
    let seg = f.segment.as_str();

    // The fit's own source, unchanged, on both stored-fit `--draws` forms and
    // through a fit.toml handle naming the stored run.
    for rest in [
        &["--draws", "posterior", "--fit", seg, "--n-draws", "2"][..],
        &["--draws", "posterior", "--fit", "fit.toml", "--n-draws", "2"][..],
        &["--draws", "d.tsv", "--fit", seg][..],
    ] {
        let (ok, err) = simulate(&f, "model.camdl", rest);
        assert!(ok, "{rest:?}: the fit's own anchored source must replay:\n{err}");
        assert!(!err.contains("is not the model this fit ran on"), "{rest:?}:\n{err}");
    }

    // `fit predict --quantities` compiles a vocabulary against the same
    // source and compares it with the archived model the same way.
    std::fs::write(f.dir.join("vocab.camdl"), "quantities {\n  peak_I = max(I)\n}\n").unwrap();
    let out = run(&f.dir, &["fit", "predict", seg, "--horizon", "free_forward", "--quantities", "vocab.camdl"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fit predict --quantities on an anchored fit:\n{err}");

    // A different anchored model is still refused, and is not called an
    // edit of the fit's source.
    let (ok, err) = simulate(&f, "edited.camdl", &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
    assert!(!ok && err.contains("edited.camdl is not the model this fit ran on"), "{err}");
    assert!(!err.contains("edited since"), "{err}");

    // The data growing after the fit moves `last_obs`, so the source,
    // byte-identical to the archived copy, now resolves to a later horizon
    // than the fit ran to. Refused, and the message says it is the data.
    let data = f.dir.join("weekly_cases.tsv");
    std::fs::write(&data, format!("{DATA}63\t30\n")).unwrap();
    let (ok, err) = simulate(&f, "model.camdl", &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
    assert!(!ok, "a horizon moved by new data is not the fitted model:\n{err}");
    assert!(err.contains("byte-identical to the source this fit archived"), "{err}");
    assert!(err.contains("observation data has changed since the fit ran"), "{err}");
    assert!(err.contains("runs to t = 77 where the fitted model ran to t = 70"), "{err}");
    assert!(!err.contains("edited since"), "{err}");
    std::fs::write(&data, DATA).unwrap();

    // The fit's own source, edited in place after the fit: refused, and named
    // as edited since — its bytes differ from the archived copy.
    std::fs::write(f.dir.join("model.camdl"), anchored.replace("@ gamma * I", "@ 2 * gamma * I")).unwrap();
    let (ok, err) = simulate(&f, "model.camdl", &["--draws", "posterior", "--fit", seg, "--n-draws", "2"]);
    assert!(!ok && err.contains("model.camdl is not the model this fit ran on"), "{err}");
    assert!(err.contains("edited since"), "{err}");

    let _ = std::fs::remove_dir_all(&f.dir);
}
