//! `observations.<stream>` quantity source (v1.1): a quantity reduces the
//! simulated observation series (`y_sim`) a run drew, not latent state.
//!
//! The load-bearing invariant pinned here is **no redraw**: an
//! `observations.<stream>` reduction folds the *same* per-draw `y_sim` the run
//! published. We assert it directly — `max(observations.reports)` from
//! `--quantities-out` must equal the maximum of the `reports` column that
//! `--obs` emits in the same run (same seed ⇒ same draws).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_camdl"))
}

/// An SIR with a weekly Poisson `reports` stream, plus obs-source quantities.
/// Params + seed are fixed so the obs draws are deterministic.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }

parameters {
  beta  : positive in [0.01, 5.0]
  gamma : positive in [0.01, 1.0]
  rho   : positive in [0.01, 1.0]
  thr   : count    in [1, 100000]
}

transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  reports {
    columns       { time : time, n : count }
    projected     = prevalence(I)
    emit_schedule = every 7 'days
    n             ~ poisson(rate = rho * projected)
  }
}

quantities {
  peak_reports  = max(observations.reports)
  total_reports = integral(observations.reports)
  onset_week    = first_above(observations.reports, thr)
}

simulate { from = 0  to = 70 }

scenarios {
  baseline {
    label = "obs"
    set = { beta = 0.3  gamma = 0.1  rho = 0.5  thr = 20 }
  }
}
"#;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn scalar_f(qdir: &Path, name: &str) -> f64 {
    let txt = std::fs::read_to_string(qdir.join("quantities").join(format!("{name}.tsv")))
        .unwrap_or_else(|e| panic!("read {name}.tsv: {e}"));
    let mut lines = txt.lines();
    // This fixture runs with `--scenario baseline`, so it has a scenario axis
    // and the design coordinate leads the row (gh#562).
    assert_eq!(
        lines.next(),
        Some("scenario\tvalue"),
        "{name}: point scalar header carries the scenario coordinate"
    );
    lines
        .next()
        .unwrap_or_else(|| panic!("{name}: missing value row"))
        .rsplit('\t')
        .next()
        .expect("value field")
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{name}: parse value: {e}"))
}

/// The emitted y_sim values for `stream` from a `--obs` TSV. The format is wide
/// — `time` then one column per stream named after the stream.
fn obs_values(obs_path: &Path, stream: &str) -> Vec<f64> {
    let txt = std::fs::read_to_string(obs_path).expect("read obs tsv");
    let mut lines = txt.lines();
    let header: Vec<&str> = lines.next().expect("obs header").split('\t').collect();
    let vcol = header
        .iter()
        .position(|c| *c == stream)
        .unwrap_or_else(|| panic!("obs tsv has no `{stream}` column; header = {header:?}"));
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            f[vcol].parse::<f64>().unwrap_or_else(|e| panic!("parse obs value `{}`: {e}", f[vcol]))
        })
        .collect()
}

#[test]
fn observation_reductions_reduce_the_same_y_sim_as_obs() {
    let tmp = tempfile::tempdir().unwrap();
    let model_path = tmp.path().join("obs_q.camdl");
    std::fs::write(&model_path, MODEL).unwrap();
    let qdir = tmp.path().join("q");
    let obs_path = tmp.path().join("obs.tsv");

    // One run emits BOTH the obs series and the quantities — same trajectory,
    // same obs_seed — so the quantity reduces exactly what `--obs` writes.
    let out = run(&[
        "simulate",
        model_path.to_str().unwrap(),
        "--scenario",
        "baseline",
        "--backend",
        "chain_binomial",
        "--seed",
        "1",
        "--obs",
        obs_path.to_str().unwrap(),
        "--quantities-out",
        qdir.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "simulate failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let emitted = obs_values(&obs_path, "reports");
    assert!(!emitted.is_empty(), "no obs rows emitted");

    // No redraw: the obs-source reductions fold the emitted y_sim exactly.
    let want_max = emitted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let peak = scalar_f(&qdir, "peak_reports");
    assert_eq!(
        peak, want_max,
        "max(observations.reports) must equal the max of the emitted --obs series (no redraw)",
    );

    // integral(observations.reports): trapezoid over the weekly obs times. The
    // emit schedule is every 7 days from t=0, so the step is a constant 7.
    let want_integral: f64 =
        emitted.windows(2).map(|w| 0.5 * (w[0] + w[1]) * 7.0).sum();
    let total = scalar_f(&qdir, "total_reports");
    assert!(
        (total - want_integral).abs() < 1e-6,
        "integral mismatch: quantity={total} hand-computed={want_integral}",
    );

    // first_above(observations.reports, thr): the first obs TIME whose y_sim
    // exceeds the threshold — a time on the weekly grid, not a snapshot time.
    let thr = 20.0;
    let want_onset = emitted
        .iter()
        .position(|&v| v > thr)
        .map(|i| (i as f64) * 7.0);
    let onset = scalar_f(&qdir, "onset_week");
    match want_onset {
        Some(t) => assert_eq!(onset, t, "first_above onset time mismatch"),
        // never crossed — the reduction right-censors to NaN.
        None => assert!(onset.is_nan(), "onset should be censored (NaN) when never crossed"),
    }
}
