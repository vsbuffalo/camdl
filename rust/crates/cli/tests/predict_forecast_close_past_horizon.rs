//! `fit predict` forecasts through the declared horizon even when the last
//! forecast row's period closes past it.
//!
//! A row's predictive value is the difference of the recorded cumulative flow
//! at its two period boundaries, so both boundaries must be recorded trajectory
//! output times (`project_coverages`). Under a closing `covers` form the last
//! forecast row's *close* sits past the label: `ending_on(time, 7 'days)` says
//! the row labelled `D` covers `[D−6, D+1)`, and the forecast tail runs out to
//! the horizon, so the final row closes one day beyond it — a day nothing was
//! integrated to. `fit predict` used to refuse the whole free-forward artifact
//! there ("the period boundary t = … is not a recorded output time"), after the
//! fit had already run; the only fix available was to move `simulate { to }` by
//! a day, which re-keys the model and orphans the fit.
//!
//! Predict integrates its own trajectories per draw, so it decides where they
//! stop: it asks for a snapshot at that close and reports the declared horizon
//! everywhere else. These tests pin the three halves of that — the row is
//! emitted, the windowed form (whose tail closes at its label) asks for
//! nothing, and the `quantities/` sidecar still stops at `simulate { to }`.
//!
//! ## The oracle
//!
//! One-way decay `A --> B @ mu * A` with `mu * t_end ≈ 0.2`, so the flow per
//! week is very nearly constant (70 falling to 60 over the record). Every
//! forecast row is a 7-day bin of that flow, so the last row must sit with its
//! neighbours; a row that had lost a day of its window, or opened at the model
//! origin, would not. No fitted parameter enters the comparison.

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
    let p = std::env::temp_dir()
        .join(format!("camdl_fcast_close_{}_{}_{}", tag, std::process::id(), ns));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// `results/fits/<stem>-<hash>/<sub>/<name>.tsv`.
fn find_artifact(root: &Path, sub: &str, name: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(root.join("fits")).ok()?.flatten() {
        let p = e.path().join(sub).join(format!("{name}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// `(time, q50)` for every row of a banded TSV on the given horizon. `horizon`
/// is `None` for the `quantities/` sidecar, which carries no such column.
fn q50_by_time(tsv: &str, horizon: Option<&str>) -> Vec<(f64, f64)> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ix = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("column {name} in header {header:?}"))
    };
    let (c_time, c_q50) = (ix("time"), ix("q50"));
    let c_hor = horizon.map(|_| ix("horizon"));
    let mut out = Vec::new();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        if let (Some(i), Some(want)) = (c_hor, horizon) {
            if c[i] != want {
                continue;
            }
        }
        out.push((c[c_time].parse::<f64>().unwrap(), c[c_q50].parse::<f64>().unwrap()));
    }
    out
}

fn at(rows: &[(f64, f64)], t: f64) -> f64 {
    rows.iter()
        .find(|(rt, _)| (*rt - t).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no row at t = {t}; rows: {rows:?}"))
        .1
}

// ── The models ──────────────────────────────────────────────────────────────

/// Weekly rows labelled by the last day they cover, the reported shape:
/// `ending_on(time, 7 'days)` on labels 7, 14, …, 161, an
/// observation-anchored horizon eight weeks past the last of them (t = 217),
/// and a daily trajectory grid. The forecast tail continues the weekly cadence
/// to 217, and that row covers `[211, 218)`.
const MODEL_ENDING_ON: &str = r#"
time_unit = 'days

compartments { A, B }

parameters {
  mu : rate  in [0.0005, 0.002] ~ log_normal(mu = -6.9, sigma = 1.0)
  N0 : count
}

transitions {
  flow : A --> B  @ mu * A
}

init { A = N0  B = 0 }

simulate {
  from = 0 'days
  to   = last_obs + 8 'weeks
  dt   = 1 'days
}

output { trajectories { every = 1 'days } }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = ending_on(time, 7 'days)
    projected     = incidence(flow)
    emit_schedule = every 7 'days
    cases ~ poisson(rate = projected)
  }
}
"#;

/// The same periods stated as `window_start`/`window_stop` columns instead.
/// The stop is the stream's time axis, so the labels are 8, 15, …, 162 and the
/// horizon is 218; the forecast tail has no rule to extrapolate and takes the
/// contiguous reading `[previous stop, label)`, which closes at its label and
/// so is always on the grid.
const MODEL_WINDOWED: &str = r#"
time_unit = 'days

compartments { A, B }

parameters {
  mu : rate  in [0.0005, 0.002] ~ log_normal(mu = -6.9, sigma = 1.0)
  N0 : count
}

transitions {
  flow : A --> B  @ mu * A
}

init { A = N0  B = 0 }

simulate {
  from = 0 'days
  to   = last_obs + 8 'weeks
  dt   = 1 'days
}

output { trajectories { every = 1 'days } }

observations {
  cases {
    columns       { win_start : window_start, win_stop : window_stop,
                    cases : count }
    projected     = incidence(flow)
    emit_schedule = every 7 'days
    cases ~ poisson(rate = projected)
  }
}
"#;

/// `MODEL_ENDING_ON` plus a series quantity and a reduction over it. Both are
/// reported over the window `simulate { to }` declares, never over the day the
/// emission grid was widened by.
fn model_with_quantities() -> String {
    MODEL_ENDING_ON.replace(
        "observations {",
        "quantities {\n  cum_b   = B\n  final_b = final(B)\n}\n\nobservations {",
    )
}

/// Flow over each `[L−6, L+1)` at mu = 1e-3, N0 = 10000:
/// `N0 (e^{−mu·(L−6)} − e^{−mu·(L+1)})`.
const CASES: &str = "time\tcases\n7\t70\n14\t69\n21\t69\n28\t68\n35\t68\n42\t67\n49\t67\n\
56\t66\n63\t66\n70\t65\n77\t65\n84\t65\n91\t64\n98\t64\n105\t63\n112\t63\n119\t62\n\
126\t62\n133\t61\n140\t61\n147\t61\n154\t60\n161\t60\n";

/// The same counts over the same periods, written as explicit windows.
const CASES_WINDOWED: &str = "win_start\twin_stop\tcases\n1\t8\t70\n8\t15\t69\n15\t22\t69\n\
22\t29\t68\n29\t36\t68\n36\t43\t67\n43\t50\t67\n50\t57\t66\n57\t64\t66\n64\t71\t65\n\
71\t78\t65\n78\t85\t65\n85\t92\t64\n92\t99\t64\n99\t106\t63\n106\t113\t63\n113\t120\t62\n\
120\t127\t62\n127\t134\t61\n134\t141\t61\n141\t148\t61\n148\t155\t60\n155\t162\t60\n";

/// A chain-binomial fit — the backend the case was reported on, and the one
/// whose output times must land on the `dt` grid.
const FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
mu = { bounds = [0.0005, 0.002], start = 0.001 }

[fixed]
N0 = 10000

[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 200
iterations = 300
burn_in = 100
thin = 1
"#;

fn setup(tag: &str, model: &str, cases: &str) -> PathBuf {
    let tmp = tempdir(tag);
    std::fs::write(tmp.join("model.camdl"), model).unwrap();
    std::fs::write(tmp.join("cases.tsv"), cases).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();
    tmp
}

fn fit_then_predict(bin: &Path, tmp: &Path, extra: &[&str]) -> std::process::Output {
    let out = run(bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut args =
        vec!["fit", "predict", "--fit", "fit.toml", "--seed", "1", "--n-draws", "40"];
    args.extend_from_slice(extra);
    run(bin, tmp, &args)
}

// ── The headline: the last forecast row is emitted, not refused ─────────────

#[test]
fn a_row_closing_past_the_horizon_is_forecast_not_refused() {
    let bin = skip_if_missing_binary();
    let tmp = setup("ending_on", MODEL_ENDING_ON, CASES);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the last forecast row closes one day past the horizon; predict must \
         integrate to that close rather than refuse the artifact.\nstdout={}\n\
         stderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !stderr.contains("not a recorded output time"),
        "the boundary refusal must be gone, not merely survived:\n{stderr}"
    );

    // The extension is announced, naming the stream, the row, and the day the
    // run integrated to — a widened grid is never silent.
    assert!(
        stderr.contains("stream 'cases'")
            && stderr.contains("217")
            && stderr.contains("Integrating to t = 218"),
        "predict must say which stream it widened the grid for, and to when:\n{stderr}"
    );

    let cases_txt = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "predictive", "cases").expect("predictive/cases.tsv"),
    )
    .unwrap();
    let ff = q50_by_time(&cases_txt, Some("free_forward"));

    // All eight forecast weeks are there, out to the horizon.
    for t in [168.0, 175.0, 182.0, 189.0, 196.0, 203.0, 210.0, 217.0] {
        assert!(
            ff.iter().any(|(rt, _)| (rt - t).abs() < 1e-9),
            "no free_forward row at t = {t}; rows: {ff:?}\n{cases_txt}"
        );
    }

    // And the last one is a whole 7-day bin like its neighbour, not a partial
    // window and not an accumulation from the origin.
    let ratio = at(&ff, 217.0) / at(&ff, 210.0);
    assert!(
        (0.75..1.3).contains(&ratio),
        "the row closing at 218 must carry the same 7 days of flow as the row \
         before it: q50(217)/q50(210) = {ratio:.3}\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The other declaration: a windowed tail asks for nothing ─────────────────

/// A `window_start`/`window_stop` stream's forecast tail takes the contiguous
/// reading `[previous stop, label)`, so it closes at its label — always a
/// recorded output time. Nothing is required, nothing is widened, and the run
/// is the one it was before the widening existed. (The strict no-op on the
/// empty request is pinned at the unit level:
/// `util::required_output_times_tests::no_required_time_leaves_the_schedule_and_the_horizon_untouched`.)
#[test]
fn a_windowed_tail_closes_at_its_label_and_widens_nothing() {
    let bin = skip_if_missing_binary();
    let tmp = setup("windowed", MODEL_WINDOWED, CASES_WINDOWED);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        !stderr.contains("Integrating to"),
        "a windowed tail closes at its label, so no grid widening may be \
         requested for it:\n{stderr}"
    );

    let cases_txt = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "predictive", "cases").expect("predictive/cases.tsv"),
    )
    .unwrap();
    let ff = q50_by_time(&cases_txt, Some("free_forward"));
    let last = ff.iter().map(|(t, _)| *t).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        last, 218.0,
        "the forecast runs to the declared horizon (last_obs 162 + 8 weeks) and \
         no further; rows: {ff:?}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The sidecar keeps the declared horizon ──────────────────────────────────

/// The widening is bookkeeping for the observation projection. `quantities {}`
/// is reported over the window `simulate { to }` declares, so a series must not
/// gain a row past it and a reduction must not fold a day the model does not
/// declare.
#[test]
fn the_quantities_sidecar_stops_at_the_declared_horizon() {
    let bin = skip_if_missing_binary();
    let tmp = setup("quantities", &model_with_quantities(), CASES);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let series = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "quantities", "cum_b").expect("quantities/cum_b.tsv"),
    )
    .unwrap();
    let rows = q50_by_time(&series, None);
    let last = rows.iter().map(|(t, _)| *t).fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        last, 217.0,
        "the series quantity is emitted on the declared window [0, 217], not on \
         the widened emission grid; rows end at {last}"
    );

    // The reduction folds the same window: `final(B)` is the series' own last
    // value, so a reduction over the widened grid would disagree with the row
    // above it.
    let scalar = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "quantities", "final_b")
            .expect("quantities/final_b.tsv"),
    )
    .unwrap();
    let mut lines = scalar.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let c_q50 = header.iter().position(|h| *h == "q50").expect("q50 column");
    let final_b: f64 =
        lines.next().expect("one row").split('\t').nth(c_q50).unwrap().parse().unwrap();
    assert!(
        (final_b - at(&rows, 217.0)).abs() < 1e-6,
        "final(B) = {final_b} must equal the series at the declared horizon \
         ({}), not a value read one day later",
        at(&rows, 217.0)
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
