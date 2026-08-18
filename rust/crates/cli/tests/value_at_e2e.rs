//! gh#538 / proposal 2026-08-17 — `value_at` end-to-end through the real
//! binary, on the two surfaces that resolve (or refuse) the `last_obs` anchor:
//!
//!   1. `fit predict` resolves `last_obs` from the fit's bound data and bands
//!      the quantity (n_censored = 0);
//!   2. a `value_at` anchored past the horizon censors EVERY draw (the
//!      censor-not-clamp contract, end-to-end — clamping would silently report
//!      the projection at t_end, the misreading `value_at` exists to prevent);
//!   3. `simulate --quantities-out` on a `last_obs` model hard-errors naming
//!      the quantity (a forward run has no observed data to anchor to).
//!
//! The `.camdl` source is compiled in-process by the `camdlc` the gate puts on
//! PATH (`make test-rust` prepends the freshly-built compiler).

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

/// SIR with a weekly case stream, cumulative-infection quantities anchored at
/// `last_obs` and at a date past the horizon. Origin 2020-03-01, t_end = 40;
/// the data (below) ends at 2020-03-29 (t = 28), so `last_obs` = 28 and the
/// `beyond` anchor (2020-05-10, t = 70) is out-of-window for every draw.
fn write_model(dir: &Path) -> PathBuf {
    let body = r#"time_unit = 'days
origin = date("2020-03-01")

compartments { S, I, R }

parameters {
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
  k     : real in [0.5, 100.0]
}

let cum_inc = I + R

transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10 }

observations {
  cases {
    columns       { time : time, cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  #' Cumulative infections at the end of observed data.
  outbreak_size = value_at(cum_inc, last_obs)
  #' Anchored past t_end = 40 — must censor every draw, never clamp.
  beyond        = value_at(cum_inc, date("2020-05-10"))
}

simulate { from = 0  to = 40 }
"#;
    let p = dir.join("value_at_e2e.camdl");
    std::fs::write(&p, body).unwrap();
    p
}

fn write_data(dir: &Path) -> PathBuf {
    let body = "time\tcases\n\
        2020-03-08\t6\n2020-03-15\t14\n2020-03-22\t9\n2020-03-29\t4\n";
    let p = dir.join("cases.tsv");
    std::fs::write(&p, body).unwrap();
    p
}

fn write_fit_toml(dir: &Path, model: &Path, data: &Path) -> PathBuf {
    let body = format!(
        r#"output_dir = "{out}"
condition_from = "first_obs - 1 week"

[model]
camdl = "{model}"

[data.observations]
cases = "{data}"

[estimate]
beta  = {{ bounds = [0.05, 2.0], start = 0.4, prior = {{ uniform = {{}} }} }}
gamma = {{ bounds = [0.01, 1.0], start = 0.2, prior = {{ uniform = {{}} }} }}

[fixed]
rho = 0.6
k = 10.0

[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 80
burn_in = 20
thin = 5
"#,
        out = dir.join("out").display(),
        model = model.display(),
        data = data.display(),
    );
    let p = dir.join("fit.toml");
    std::fs::write(&p, body).unwrap();
    p
}

/// The stored fit segment dir (`fits/<stem>-<h8>/`) under `out/`.
fn fit_segment(out: &Path) -> PathBuf {
    let fits = out.join("fits");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&fits)
        .unwrap_or_else(|e| panic!("no fits dir {}: {e}", fits.display()))
        .filter_map(|d| d.ok().map(|d| d.path()))
        .filter(|p| p.is_dir())
        .collect();
    assert_eq!(dirs.len(), 1, "expected one fit segment, got {dirs:?}");
    dirs.pop().unwrap()
}

#[test]
fn predict_resolves_last_obs_and_censors_past_horizon() {
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    let data = write_data(tmp.path());
    let fit_toml = write_fit_toml(tmp.path(), &model, &data);

    let fit = Command::new(bin())
        .args(["fit", "run"])
        .arg(&fit_toml)
        .arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("fit run");
    assert!(
        fit.status.success(),
        "fit run failed:\n{}",
        String::from_utf8_lossy(&fit.stderr)
    );

    let segment = fit_segment(&tmp.path().join("out"));
    let predict = Command::new(bin())
        .args(["fit", "predict"])
        .arg(&segment)
        .arg("--no-progress")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("fit predict");
    assert!(
        predict.status.success(),
        "fit predict failed:\n{}",
        String::from_utf8_lossy(&predict.stderr)
    );

    // The manifest declares both value_at quantities censorable; the banded
    // TSVs carry the censoring trio. `outbreak_size` (last_obs = 28, inside
    // [0, 40]) bands every draw; `beyond` (t = 70) censors every draw.
    let manifest_path = segment.join("quantities.json");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("no quantities.json at {}: {e}", manifest_path.display())),
    )
    .expect("parse quantities.json");
    let entries = manifest["quantities"].as_array().expect("quantities array");
    for name in ["outbreak_size", "beyond"] {
        let e = entries
            .iter()
            .find(|e| e["name"] == name)
            .unwrap_or_else(|| panic!("{name} missing from manifest: {entries:?}"));
        assert_eq!(e["reduce"], "value_at", "{e}");
        assert!(
            e["censoring"].is_object(),
            "a value_at scalar is censorable and must say so in the manifest: {e}"
        );
    }

    // The censoring trio, from the banded TSVs.
    let trio = |name: &str| -> (u64, u64) {
        let path = segment.join("quantities").join(format!("{name}.tsv"));
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("no {}: {e}", path.display()));
        let mut lines = body.lines();
        let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
        let row: Vec<&str> = lines.next().expect("one data row").split('\t').collect();
        let col = |c: &str| -> u64 {
            let i = header
                .iter()
                .position(|h| *h == c)
                .unwrap_or_else(|| panic!("{c} missing from {header:?}"));
            row[i].parse().unwrap_or_else(|e| panic!("{c}={:?}: {e}", row[i]))
        };
        (col("n_value"), col("n_censored"))
    };

    let (n_v, n_c) = trio("outbreak_size");
    assert!(n_v > 0 && n_c == 0, "last_obs=28 is in-window: n_value={n_v} n_censored={n_c}");

    let (n_v, n_c) = trio("beyond");
    assert!(
        n_v == 0 && n_c > 0,
        "an anchor past t_end must censor EVERY draw (clamping would report \
         final()): n_value={n_v} n_censored={n_c}"
    );
}

#[test]
fn simulate_refuses_last_obs_naming_the_quantity() {
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path());
    std::fs::write(
        tmp.path().join("params.toml"),
        "beta = 0.4\ngamma = 0.2\nrho = 0.6\nk = 10.0\n",
    )
    .unwrap();

    let out = Command::new(bin())
        .arg("simulate")
        .arg(&model)
        .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
        .args(["--params", tmp.path().join("params.toml").to_str().unwrap()])
        .args(["-o", tmp.path().join("traj.tsv").to_str().unwrap()])
        .arg("--quantities-out")
        .arg(tmp.path().join("q"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("simulate");
    assert!(!out.status.success(), "simulate must refuse a last_obs model");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("outbreak_size") && stderr.contains("last_obs"),
        "the error must name the quantity and the anchor: {stderr}"
    );
    assert!(
        stderr.contains("fit predict"),
        "the error must point at the surface that CAN resolve it: {stderr}"
    );
}
