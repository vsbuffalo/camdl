//! A time function evaluated outside the range on which it is defined is an
//! error unless the run is given `--allow-forcing-extrapolation` (or, in a fit
//! config, `[config] allow_forcing_extrapolation = true`). Under the override
//! the end value is held exactly as before, one warning per function names its
//! range and the span held, and the override is recorded in `run.json`.
//!
//! One model per kind that has a range: an `interpolated` forcing read
//! linearly, as a cubic spline and piecewise-constant (`method = linear |
//! spline | constant`), and a `bspline` over a window. Each forcing's data, or
//! window, ends at t = 50 while the run goes to t = 100.
//!
//! Spec: docs/dev/temporal-architecture.md §4.6 (D44, D48); proposal
//! docs/dev/proposals/2026-11-15-temporal-architecture-migration.md Stage 0(f).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn bin() -> PathBuf {
    let p = repo().join("rust/target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build`", p.display());
    p
}

/// The forcing block of each kind; `end` is the last data point or the
/// window end.
fn forcing(kind: &str, end: u32) -> String {
    match kind {
        "linear" | "spline" | "constant" => format!(
            "  mult : interpolated 'ratio {{\n    data = \"mult.tsv\"\n    time_col = \"time\"\n    \
             value_col = \"mult\"\n    method = {kind}\n  }}"
        ),
        "bspline" => format!(
            "  mult : bspline 'ratio {{\n    window = [0 'days, {end} 'days]\n    n_basis = 4\n    \
             coefs = [1.0, 0.8, 0.6, 0.5]\n  }}"
        ),
        other => panic!("no kind {other}"),
    }
}

const KINDS: [&str; 4] = ["linear", "spline", "constant", "bspline"];

/// An SIR model whose infection rate is scaled by `mult`, observed weekly,
/// run from t = 0 to t = 100. For the data-table kinds, `mult.tsv` holds one
/// row per `rows` entry.
fn model(dir: &Path, kind: &str, rows: &[(u32, f64)]) -> PathBuf {
    let mut tsv = String::from("time\tmult\n");
    for (t, v) in rows {
        tsv.push_str(&format!("{t}\t{v}\n"));
    }
    std::fs::write(dir.join("mult.tsv"), tsv).unwrap();
    let end = rows.last().map_or(50, |r| r.0);
    let p = dir.join(format!("{kind}.camdl"));
    std::fs::write(
        &p,
        format!(
            r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
}}

let N = S + I + R

forcing {{
{forcing}
}}

transitions {{
  infection : S --> I @ mult * beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990  I = 10 }}

observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }}
}}

simulate {{ from = 0 'days  to = 100 'days }}
"#,
            forcing = forcing(kind, end)
        ),
    )
    .unwrap();
    p
}

/// Data from t = 0 to t = 50, weekly-ish, ending on a value other than its
/// neighbours so a hold is visible.
const ROWS_TO_50: [(u32, f64); 6] = [(0, 1.0), (10, 0.9), (20, 0.7), (30, 0.8), (40, 0.6), (50, 0.5)];

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .env_remove("CLICOLOR_FORCE")
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        .env("CAMDL_OUTPUT_DIR", dir.join("cas"))
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// `camdl simulate` of `model`, its trajectory mirrored to `traj.tsv`.
fn simulate(dir: &Path, model: &Path, extra: &[&str]) -> Output {
    std::fs::write(dir.join("params.toml"), "beta = 0.3\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let traj = dir.join("traj.tsv");
    let params = dir.join("params.toml");
    let mut args = vec![
        "simulate",
        model.to_str().unwrap(),
        "--backend",
        "chain_binomial",
        "--seed",
        "1",
        "--dt",
        "1",
        "--params",
        params.to_str().unwrap(),
        "-o",
        traj.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    camdl(dir, &args)
}

/// A tiny PGAS `fit run` of `model` on weekly counts over days 7–70, past the
/// forcing's range, with `config_extra` appended to `[config]`.
fn fit_run(dir: &Path, model: &Path, config_extra: &str) -> Output {
    let data = dir.join("cases.tsv");
    let mut body = String::from("time\tcases\n");
    for w in 1..=10 {
        body.push_str(&format!("{}\t{}\n", 7 * w, 10 + 3 * w));
    }
    std::fs::write(&data, body).unwrap();
    let toml = dir.join("fit.toml");
    std::fs::write(
        &toml,
        format!(
            "output_dir = \"{out}\"\n[model]\ncamdl = \"{model}\"\n[data.observations]\ncases = \"{data}\"\n\
             [config]\ndt = 1.0\n{config_extra}[estimate]\n\
             gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ uniform = {{}} }} }}\n\
             [fixed]\nbeta = 0.3\nrho = 0.6\n\
             [method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\n\
             sweeps = 4\nburn_in = 2\n",
            out = dir.join("results").display(),
            model = model.display(),
            data = data.display(),
        ),
    )
    .unwrap();
    camdl(dir, &["fit", "run", toml.to_str().unwrap()])
}

/// The one line of the warning a held forcing gets, by its code.
const WARNING: &str = "forcing-outside-range";

fn warnings(stderr: &str) -> usize {
    stderr.matches("forcing `mult` is defined on").count()
}

/// Each kind, past its range under `simulate`, is refused: a non-zero exit
/// naming the forcing, its range, the first time outside it and the override.
#[test]
fn simulate_refuses_each_kind_past_its_range() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = simulate(tmp.path(), &m, &[]);
        let err = stderr(&o);
        assert!(!o.status.success(), "{kind}: a run past the forcing's range must be refused:\n{err}");
        assert!(err.contains("forcing `mult`"), "{kind}: names the forcing:\n{err}");
        assert!(err.contains("t = 0 to t = 50"), "{kind}: names the range:\n{err}");
        assert!(err.contains("at t = 51"), "{kind}: names the first time outside it:\n{err}");
        assert!(err.contains("--allow-forcing-extrapolation"), "{kind}: names the override:\n{err}");
        assert!(!tmp.path().join("traj.tsv").exists(), "{kind}: a refused run writes no trajectory");
    }
}

/// The same under `fit run`, and `[config] allow_forcing_extrapolation = true`
/// lets the fit run, with one warning.
#[test]
fn fit_run_refuses_each_kind_past_its_range() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = fit_run(tmp.path(), &m, "");
        let err = stderr(&o);
        assert!(!o.status.success(), "{kind}: fit run past the forcing's range must be refused:\n{err}");
        assert!(err.contains("forcing `mult`") && err.contains("--allow-forcing-extrapolation"), "{kind}:\n{err}");

        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = fit_run(tmp.path(), &m, "allow_forcing_extrapolation = true\n");
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: the config override lets the fit run:\n{err}");
        assert_eq!(warnings(&err), 1, "{kind}: one warning for the one forcing:\n{err}");
    }
}

/// Under `--allow-forcing-extrapolation` each kind runs, says so once, and
/// records the override in the run's `run.json`.
#[test]
fn the_override_holds_with_one_warning_per_function() {
    for kind in KINDS {
        let tmp = tempfile::tempdir().unwrap();
        let m = model(tmp.path(), kind, &ROWS_TO_50);
        let o = simulate(tmp.path(), &m, &["--allow-forcing-extrapolation"]);
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: the override lets the run proceed:\n{err}");
        assert_eq!(warnings(&err), 1, "{kind}: exactly one warning:\n{err}");
        assert!(err.contains(WARNING) || err.contains("warning"), "{kind}: it is a warning:\n{err}");
        assert!(err.contains("held from t = 50 to the end of the run"), "{kind}: names the span held:\n{err}");
        let recorded = recorded_overrides(&tmp.path().join("cas"));
        assert_eq!(recorded, vec![true], "{kind}: the override is recorded in run.json");
    }
}

/// The held values are the values the forcing had before this check existed:
/// for the linear and piecewise-constant reads, a hold at the last data point
/// is the same function as data extended by one more row carrying the last
/// value, so the two trajectories are byte-identical. (A cubic spline or a
/// B-spline changes shape inside its range when a point is added, so they
/// have no such oracle; the hold itself is untouched code.)
#[test]
fn the_held_trajectory_is_the_extended_data_trajectory() {
    for kind in ["linear", "constant"] {
        let held = tempfile::tempdir().unwrap();
        let m = model(held.path(), kind, &ROWS_TO_50);
        let o = simulate(held.path(), &m, &["--allow-forcing-extrapolation"]);
        assert!(o.status.success(), "{kind}:\n{}", stderr(&o));

        let extended = tempfile::tempdir().unwrap();
        let mut rows = ROWS_TO_50.to_vec();
        rows.push((100, 0.5));
        let m = model(extended.path(), kind, &rows);
        let o = simulate(extended.path(), &m, &[]);
        let err = stderr(&o);
        assert!(o.status.success(), "{kind}: data covering the run need no override:\n{err}");
        assert_eq!(warnings(&err), 0, "{kind}: no warning in range:\n{err}");

        let a = std::fs::read(held.path().join("traj.tsv")).unwrap();
        let b = std::fs::read(extended.path().join("traj.tsv")).unwrap();
        assert!(a == b, "{kind}: the held trajectory must equal the extended-data one byte for byte");
    }
}

/// A stored run made under the override does not answer for the same run
/// without it: the second run is refused, not served the held values.
#[test]
fn a_stored_held_run_is_not_served_without_the_override() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let o = simulate(tmp.path(), &m, &["--allow-forcing-extrapolation"]);
    assert!(o.status.success(), "{}", stderr(&o));
    std::fs::remove_file(tmp.path().join("traj.tsv")).unwrap();
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(!o.status.success(), "the stored held run must not be served:\n{err}");
    assert!(err.contains("--allow-forcing-extrapolation"), "{err}");
}

/// A forcing whose data start after the run does is refused too, naming the
/// side it was evaluated on.
#[test]
fn a_run_before_the_first_data_point_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &[(10, 1.0), (50, 0.8), (100, 0.6)]);
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(!o.status.success(), "{err}");
    assert!(err.contains("before its first data point") && err.contains("at t = 0"), "{err}");
}

/// The check reads the times evaluated, not the run's span: a forcing read
/// only under a condition that is false past its range raises nothing.
#[test]
fn a_forcing_read_only_inside_its_range_is_not_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "linear", &ROWS_TO_50);
    let src = std::fs::read_to_string(&m).unwrap().replace(
        "@ mult * beta * S * (I / N)",
        "@ (if t < 40 'days then mult else 1.0) * beta * S * (I / N)",
    );
    assert!(src.contains("if t < 40 'days"), "the rate was rewritten");
    std::fs::write(&m, src).unwrap();
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert_eq!(warnings(&err), 0, "{err}");
}

/// A B-spline window that covers the run, and data that do, are not affected.
#[test]
fn a_range_covering_the_run_is_untouched() {
    let tmp = tempfile::tempdir().unwrap();
    let m = model(tmp.path(), "bspline", &[(0, 1.0), (100, 1.0)]);
    let o = simulate(tmp.path(), &m, &[]);
    let err = stderr(&o);
    assert!(o.status.success(), "{err}");
    assert_eq!(warnings(&err), 0, "{err}");
}

/// The `allow_forcing_extrapolation` value of every committed `run.json`
/// under `cas`.
fn recorded_overrides(cas: &Path) -> Vec<bool> {
    let mut out = Vec::new();
    let mut stack = vec![cas.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                if let Some(b) = v.pointer("/inputs/forcing_extrapolation/allowed").and_then(|b| b.as_bool()) {
                    out.push(b);
                }
            }
        }
    }
    out
}
