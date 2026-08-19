//! gh#630 (ebola F35): `simulate --draws posterior` must never silently replay
//! the full posterior cloud. `fit predict` caps at a strided 200 and says why;
//! simulate's `--n-draws` used to apply only to `--draws uniform/prior`, so a
//! 60k-draw cloud replayed in full (hours of forward solves, a 21 MB
//! --draws-out). Pinned here: `--n-draws N` caps the posterior replay to a
//! strided N (observable via --draws-out row count and the stderr line).

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
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

#[test]
fn n_draws_caps_posterior_replay_strided() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir()
        .join(format!("camdl_sim_post_cap_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    // 30 sweeps − 10 burn-in = a 20-draw cloud: small fit, real posterior.
    std::fs::write(tmp.join("fit.toml"), r#"output_dir = "results"
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
[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 1
particles = 100
sweeps = 30
burn_in = 10
thin = 1
"#).unwrap();

    let out = run(&bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));

    // Find the fit segment dir (results/fits/<segment>).
    let fits = tmp.join("results").join("fits");
    let segment = std::fs::read_dir(&fits).unwrap().flatten()
        .map(|e| e.path()).find(|p| p.is_dir())
        .expect("fit segment dir");

    // Replay with --n-draws 5: exactly 5 strided draws, said on stderr.
    let out = run(&bin, &tmp, &[
        "simulate", "model.camdl",
        "--draws", "posterior",
        "--fit", &segment.to_string_lossy(),
        "--n-draws", "5",
        "--draws-out", "picked.tsv",
        "--output", "traj_picked.tsv",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "simulate failed:\nstderr={stderr}");
    assert!(stderr.contains("subsampling 5 of 20"),
        "the cap must be announced (strided 5 of the 20-draw cloud):\n{stderr}");
    let picked = std::fs::read_to_string(tmp.join("picked.tsv")).unwrap();
    let rows = picked.lines().filter(|l| !l.trim_start().starts_with('#'))
        .count().saturating_sub(1);
    assert_eq!(rows, 5,
        "--draws-out must carry exactly the capped draw set, got {rows} rows");

    // Without --n-draws, a 20-draw cloud is UNDER the default cap (200): all
    // 20 replay, and no subsampling line prints — the cap is a ceiling, not a
    // resample.
    let out = run(&bin, &tmp, &[
        "simulate", "model.camdl",
        "--draws", "posterior",
        "--fit", &segment.to_string_lossy(),
        "--draws-out", "all.tsv",
        "--output", "traj_all.tsv",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "simulate failed:\nstderr={stderr}");
    assert!(!stderr.contains("subsampling"),
        "a cloud under the cap must not be subsampled:\n{stderr}");
    let all = std::fs::read_to_string(tmp.join("all.tsv")).unwrap();
    let rows = all.lines().filter(|l| !l.trim_start().starts_with('#'))
        .count().saturating_sub(1);
    assert_eq!(rows, 20, "under-cap cloud replays in full, got {rows}");

    let _ = std::fs::remove_dir_all(&tmp);
}
