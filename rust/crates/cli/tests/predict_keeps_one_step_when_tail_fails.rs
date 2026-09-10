//! `fit predict` keeps what it computed when the free-forward tail refuses.
//!
//! The one-step-ahead band and the free-forward tail are two different objects.
//! The first is data-conditioned — a bootstrap filter over the observed rows,
//! `p(y_t | y_{1:t-1})` — so it never reaches the forecast horizon and cannot
//! be affected by anything the horizon does. The second replays each draw
//! forward past the data. Before this, one refusal in the second took the whole
//! command down: nothing was written, including the one-step half that had
//! already finished.
//!
//! The ruling (proposal `docs/dev/proposals/2026-09-08-workflow-first-fit-config.md`
//! §3.5, §8 item 15): write the one-step artifact, record the tail failure in
//! `report.json` under `failures`, print what failed and what was written, and
//! exit 1. Fail closed on the exit status and the report, not by discarding a
//! complete object.
//!
//! The tail failure used here is a scenario whose declared `simulate { to }`
//! differs from the model's (`refuse_scenario_horizon`, gh#561): `fit predict`
//! replays every scenario at the model's own horizon, so it cannot honour that
//! window and says so. It is deterministic, cheap, and needs no pathological
//! model.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation, plus one scenario that
/// declares its own (longer) horizon. The scenario changes nothing else, so the
/// refusal under test is about the horizon and nothing but the horizon.
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

scenarios {
  longer { simulate { to = 160 'days } }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

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
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 200
sweeps = 60
burn_in = 20
thin = 1
"#;

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The fit segment: the directory under `results/fits/` holding the artifacts.
fn segment(results: &Path) -> PathBuf {
    let fits = results.join("fits");
    for e in std::fs::read_dir(&fits).expect("results/fits exists").flatten() {
        let p = e.path();
        if p.join("predictive.json").is_file() || p.join("report.json").is_file() {
            return p;
        }
    }
    panic!("no fit segment carrying a predictive artifact under {}", fits.display());
}

#[test]
fn a_refused_tail_still_writes_the_one_step_artifact_and_records_the_failure() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir()
        .join(format!("camdl_predict_tail_fail_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The scenario declares t = 160 where the model declares t = 80, so the
    // free-forward tail cannot be produced for it.
    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", "--fit", "fit.toml", "--scenario", "longer"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    // 1. Fail closed on the exit status. A caller checking `$?` is told.
    assert_eq!(
        out.status.code(),
        Some(1),
        "a deterministic failure exits 1:\nstdout={stdout}\nstderr={stderr}"
    );

    let results = tmp.join("results");
    let seg = segment(&results);

    // 2. The one-step half — complete, data-conditioned, never reaching the
    //    horizon — is on disk. This is the object the old behaviour discarded.
    let pred = seg.join("predictive").join("weekly_cases.tsv");
    assert!(
        pred.is_file(),
        "the one-step predictive must be written even though the tail refused; \
         segment holds {:?}\nstdout={stdout}\nstderr={stderr}",
        std::fs::read_dir(&seg)
            .map(|d| d.flatten().map(|e| e.file_name()).collect::<Vec<_>>())
            .unwrap_or_default()
    );
    let text = std::fs::read_to_string(&pred).unwrap();
    let horizons: Vec<&str> = text
        .lines()
        .skip(1)
        .filter_map(|l| l.split('\t').nth(2))
        .collect();
    assert!(
        horizons.contains(&"one_step"),
        "the written rows are the one-step band:\n{text}"
    );
    assert!(
        !horizons.contains(&"free_forward"),
        "no free-forward row may be written when the tail refused — a partial \
         tail reads as a complete one:\n{text}"
    );

    // 3. The failure is recorded, naming the site and carrying the reason.
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(seg.join("report.json")).expect("report.json is written"),
    )
    .expect("report.json parses");
    let failures = report["failures"].as_array().expect("failures is a list");
    assert_eq!(failures.len(), 1, "one failure recorded: {report:#}");
    assert_eq!(failures[0]["kind"], "evaluation_failed");
    assert_eq!(failures[0]["at"], "free_forward");
    assert!(
        failures[0]["reason"].as_str().unwrap().contains("longer"),
        "the reason names the scenario that refused: {report:#}"
    );

    // 4. The run says what failed and what it wrote anyway.
    assert!(
        stderr.contains("the free-forward tail"),
        "stderr names what failed:\n{stderr}"
    );
    assert!(
        stdout.lines().any(|l| l.starts_with("wrote ")),
        "stdout still lists the files written:\n{stdout}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// The other half of the rule, and the negative control for the test above:
/// `--horizon one_step` never builds a tail, so a scenario it could not have
/// honoured is a plain usage error — refused outright, with no artifact and no
/// failure record, exactly as before.
#[test]
fn an_unhonourable_scenario_is_still_a_usage_error_when_no_tail_was_asked_for() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir()
        .join(format!("camdl_predict_tail_onestep_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed: {}", String::from_utf8_lossy(&out.stderr));

    let out = run(
        &bin,
        &tmp,
        &[
            "fit", "predict", "--fit", "fit.toml", "--scenario", "longer",
            "--horizon", "one_step",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(1), "still refused: {stderr}");
    assert!(
        stderr.contains("error: scenario 'longer'"),
        "refused as a usage error, not recorded as a tail failure:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
