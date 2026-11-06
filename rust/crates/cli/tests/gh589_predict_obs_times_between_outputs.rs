//! gh#589: `fit predict` projects a stream at data times that fall between the
//! model's output times.
//!
//! The particle filter scores each row from integrator state at its data time,
//! so a record whose times (here every 2.5 days) miss the default 1-day output
//! grid fits fine. `fit predict --horizon free_forward` replays its own
//! trajectories and reads each row off the *recorded* snapshots, so it used to
//! refuse — "observation stream 'prev' needs a value at t = 2.5, which is not a
//! recorded output time" — leaving only an `output { trajectories { every } }`
//! line, which re-keys the model and orphans the fit. Predict now records a
//! snapshot at every time its projection reads; these tests pin that, and that
//! the extra snapshots stay internal (`quantities/` keeps the declared grid).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_gh589_{}_{}_{}", tag, std::process::id(), ns));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Newest `fits/<segment>/predict/<run id>/` directory holding `sub/name.tsv`.
fn find_artifact(root: &Path, sub: &str, name: &str) -> Option<PathBuf> {
    for seg in std::fs::read_dir(root.join("fits")).ok()?.flatten() {
        for d in std::fs::read_dir(seg.path().join("predict")).into_iter().flatten().flatten() {
            let p = d.path().join(sub).join(format!("{name}.tsv"));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// The `time` column of a TSV, optionally restricted to one `horizon` value.
fn times(tsv: &str, horizon: Option<&str>) -> Vec<f64> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ix = |n: &str| header.iter().position(|h| *h == n).unwrap_or_else(|| panic!("{n} in {header:?}"));
    let c_time = ix("time");
    let c_hor = horizon.map(|_| ix("horizon"));
    lines
        .filter_map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            if let (Some(i), Some(h)) = (c_hor, horizon) {
                if c[i] != h {
                    return None;
                }
            }
            Some(c[c_time].parse::<f64>().unwrap())
        })
        .collect()
}

/// An SIR model with no `output` block (so the 1-day default grid) and `dt =
/// 0.5`, fitted to a prevalence stream (a state read at an instant) and an
/// incidence stream (a flow over a window), both reported every 2.5 days.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.05, 2.0] ~ log_normal(mu = -0.7, sigma = 0.5)
  gamma : rate in [0.01, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  prev {
    columns       { time : time, prev : count }
    projected     = prevalence(I)
    prev ~ poisson(rate = projected)
  }
  inc {
    columns       { win_start : window_start, win_stop : window_stop, inc : count }
    projected     = incidence(infection)
    inc ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 20 'days  dt = 0.5 'days }
"#;

const WITH_QUANTITIES: &str = "quantities {\n  cum_r = R\n}\n\nobservations {";

const PREV: &str = "time\tprev\n2.5\t18\n5\t30\n7.5\t52\n10\t80\n12.5\t96\n15\t90\n";
const INC: &str = "win_start\twin_stop\tinc\n0\t2.5\t20\n2.5\t5\t35\n5\t7.5\t60\n7.5\t10\t85\n\
10\t12.5\t80\n12.5\t15\t70\n";

const FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
prev = "prev.tsv"
inc = "inc.tsv"

[estimate]
beta = { bounds = [0.05, 2.0], start = 0.5 }

[fixed]
gamma = 0.1

[config]
dt = 0.5

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 1
particles = 30
iterations = 60
burn_in = 20
thin = 1
"#;

fn setup(tag: &str, model: &str) -> PathBuf {
    let tmp = tempdir(tag);
    std::fs::write(tmp.join("model.camdl"), model).unwrap();
    std::fs::write(tmp.join("prev.tsv"), PREV).unwrap();
    std::fs::write(tmp.join("inc.tsv"), INC).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();
    tmp
}

fn fit_then_predict(tmp: &Path) -> std::process::Output {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let out = run(&bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "the off-grid record must fit (the filter scores from integrator state):\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    run(
        &bin,
        tmp,
        &["fit", "predict", "--fit", "fit.toml", "--seed", "1", "--n-draws", "10",
          "--horizon", "free_forward"],
    )
}

const DATA_TIMES: [f64; 6] = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0];

/// Regression guard: a record already on the output grid requests nothing, so
/// predict runs exactly as before (the same-seed byte-identity of its output
/// against the previous binary was checked by diffing the two trees).
#[test]
fn an_on_grid_record_requests_no_extra_snapshots() {
    let model = MODEL.replace(
        "simulate {",
        "output { trajectories { every = 0.5 'days } }\nsimulate {",
    );
    let tmp = setup("on_grid", &model);
    let out = fit_then_predict(&tmp);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "predict failed:\n{stderr}");
    assert!(
        !stderr.contains("recording extra trajectory snapshots"),
        "every data time is an output time, so nothing may be requested:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn predict_projects_at_data_times_between_output_times() {
    let tmp = setup("off_grid", MODEL);
    let out = fit_then_predict(&tmp);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success() && !stderr.contains("not a recorded output time"),
        "predict must record a snapshot at each data time, not refuse:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    let root = tmp.join("results");
    for stream in ["prev", "inc"] {
        let pred = std::fs::read_to_string(
            find_artifact(&root, "predictive", stream)
                .unwrap_or_else(|| panic!("predictive/{stream}.tsv")),
        )
        .unwrap();
        assert_eq!(
            times(&pred, Some("free_forward")),
            DATA_TIMES,
            "predictive/{stream}.tsv rows:\n{pred}"
        );
        let obs = std::fs::read_to_string(
            find_artifact(&root, "observed", stream)
                .unwrap_or_else(|| panic!("observed/{stream}.tsv")),
        )
        .unwrap();
        assert_eq!(times(&obs, None), DATA_TIMES, "observed/{stream}.tsv rows:\n{obs}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// The extra snapshots are bookkeeping for the projection: a `quantities {}`
/// series stays on the declared 1-day output grid, 0 through 20, with no row at
/// a data time.
#[test]
fn the_extra_snapshots_do_not_reach_the_quantities_sidecar() {
    let model = MODEL.replace("observations {", WITH_QUANTITIES);
    let tmp = setup("quantities", &model);
    let out = fit_then_predict(&tmp);
    assert!(
        out.status.success(),
        "predict failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let q = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "quantities", "cum_r").expect("quantities/cum_r.tsv"),
    )
    .unwrap();
    let want: Vec<f64> = (0..=20).map(f64::from).collect();
    assert_eq!(times(&q, None), want, "quantities/cum_r.tsv must stay on the declared grid:\n{q}");
    let _ = std::fs::remove_dir_all(&tmp);
}
