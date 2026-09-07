//! gh#702, restated under gh#833 — `fit predict` opens the first incidence bin
//! where the stream's declaration opens it, not at the model origin.
//!
//! A stream declaring `covers = closing_at(time, 50 'days)` whose first row is
//! labelled 150 covers `[100, 150)`: the fit simulates `[0, 100)` as warm-up and
//! scores the first datum over `[100, 150)`. The predictive artifact is plotted
//! against that same datum, so it must report the flow over the SAME period.
//! Before the declaration existed this needed a separate `condition_from` key,
//! and the predictive once ignored it, reporting `(0, 150]` — the whole warm-up
//! folded into the first row (gh#702). Now the declaration is the only source
//! of the window, read by the likelihood and the predictive alike.
//!
//! Only the FIRST row is affected on the free-forward horizon: every later row
//! is a period between two observed labels. On a single-observation fit the
//! first row is the only row, so the error would be the whole artifact.
//!
//! ## The oracle
//!
//! The model is a one-way decay `A --> B @ mu * A` with `mu * t_end << 1`, so
//! the flow per unit time is very nearly constant. Observations sit at t = 150
//! and t = 200, each declared to cover the 50 days closing at its label, which
//! makes the two scored bins `[100, 150)` and `[150, 200)` — equal width, equal
//! expected flow. The ratio of the two predictive medians is therefore ~1 when
//! the first bin opens at 100, and ~3 when it opens at t = 0 (a 150-day bin
//! against a 50-day one). No fitted parameter enters the ratio, so the oracle
//! does not depend on how well the fit converged.
//!
//! The prevalence stream in the same model is the cross-check for the other
//! direction: a stock reads the state at an instant and has no accumulator to
//! reset, so its t = 150 row must stay at the whole-run cumulative level —
//! ~3× the incidence row, not equal to it.

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
    let p = std::env::temp_dir().join(format!("camdl_gh702_{}_{}_{}", tag, std::process::id(), ns));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// `results/fits/<stem>-<hash>/<sub>/<stream>.tsv`.
fn find_artifact(root: &Path, sub: &str, stream: &str) -> Option<PathBuf> {
    for e in std::fs::read_dir(root.join("fits")).ok()?.flatten() {
        let p = e.path().join(sub).join(format!("{stream}.tsv"));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// `(time, q50)` for every row of a predictive TSV on the given horizon.
fn q50_by_time(tsv: &str, horizon: &str) -> Vec<(f64, f64)> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ix = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("column {name} in header {header:?}"))
    };
    let (c_time, c_hor, c_q50) = (ix("time"), ix("horizon"), ix("q50"));
    let mut out = Vec::new();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        if c[c_hor] != horizon {
            continue;
        }
        out.push((c[c_time].parse::<f64>().unwrap(), c[c_q50].parse::<f64>().unwrap()));
    }
    out
}

fn at(rows: &[(f64, f64)], t: f64) -> f64 {
    rows.iter()
        .find(|(rt, _)| (*rt - t).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no predictive row at t = {t}; rows: {rows:?}"))
        .1
}

// ── The model ───────────────────────────────────────────────────────────────

/// One-way decay from a large `A`, so the flow per day is very nearly constant
/// over the run (`mu * 200 = 0.02`). Two streams off the same dynamics:
/// `cases` is an INTERVAL (incidence) stream declaring that each row covers the
/// 50 days closing at its label — the declaration that places its first bin —
/// and `stock` is an INSTANT (prevalence) stream reading `B` itself, which has
/// no accumulator and must be untouched by any window.
///
/// `output { trajectories { every = 50 'days } }` puts the period boundaries
/// 100, 150, 200 on the recorded snapshot grid, which is what lets the
/// projection read the cumulative flow there at all.
const MODEL: &str = r#"
time_unit = 'days

compartments { A, B }

parameters {
  mu : rate  in [0.00005, 0.0002] ~ log_normal(mu = -9.0, sigma = 1.0)
  N0 : count
}

transitions {
  flow : A --> B  @ mu * A
}

init { A = N0  B = 0 }

simulate {
  from = 0 'days
  to   = 200 'days
}

output { trajectories { every = 50 'days } }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 50 'days)
    projected     = incidence(flow)
    emit_schedule = every 50 'days
    cases ~ poisson(rate = projected)
  }
  stock {
    columns       { time : time, stock : count }
    projected     = prevalence(B)
    emit_schedule = every 50 'days
    stock ~ poisson(rate = projected)
  }
}
"#;

/// Flow over [100, 150) and [150, 200) at mu = 1e-4, N0 = 1e6:
/// `N0 (e^{-mu t1} - e^{-mu t2})` = 4938 and 4913.
const CASES: &str = "time\tcases\n150\t4938\n200\t4913\n";

/// The same model declaring 25-day windows: a row labelled 150 (on the
/// `every = 50 'days` output grid) then covers `[125, 150)`, whose START is not
/// a recorded output time.
fn model_off_grid() -> String {
    MODEL.replace("closing_at(time, 50 'days)", "closing_at(time, 25 'days)")
}

/// A single row labelled 150 under the 25-day declaration: flow over
/// `[125, 150)` at mu = 1e-4, N0 = 1e6 is ~2470.
const CASES_OFF_GRID: &str = "time\tcases\n150\t2470\n";

/// `B(t) = N0 (1 - e^{-mu t})` at t = 150 and 200 — the cumulative stock, which
/// no window resets.
const STOCK: &str = "time\tstock\n150\t14888\n200\t19801\n";

/// A chain-binomial fit, so BOTH predictive horizons are reachable
/// (`--horizon one_step` needs a filterable fit).
fn fit_toml(out_dir: &str) -> String {
    format!(
        r#"output_dir = "{out_dir}"

[model]
camdl = "model.camdl"

[data.observations]
cases = "cases.tsv"
stock = "stock.tsv"

[estimate]
mu = {{ bounds = [0.00005, 0.0002], start = 0.0001 }}

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 300
burn_in = 100
thin = 1
"#
    )
}

/// An ODE fit of the same model + data — the configuration gh#702 was filed
/// on. The projection is backend-independent (it reads the recorded
/// trajectory), and this pins that.
fn ode_fit_toml(out_dir: &str) -> String {
    format!(
        r#"output_dir = "{out_dir}"

[model]
camdl = "model.camdl"

[data.observations]
cases = "cases.tsv"
stock = "stock.tsv"

[estimate]
mu = {{ bounds = [0.00005, 0.0002], start = 0.0001 }}

[fixed]
N0 = 1000000

[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 800
burn_in = 200
thin = 1
"#
    )
}

fn setup(tag: &str, toml: &str, cases: &str) -> PathBuf {
    setup_with_model(tag, toml, cases, MODEL)
}

fn setup_with_model(tag: &str, toml: &str, cases: &str, model: &str) -> PathBuf {
    let tmp = tempdir(tag);
    std::fs::write(tmp.join("model.camdl"), model).unwrap();
    std::fs::write(tmp.join("cases.tsv"), cases).unwrap();
    std::fs::write(tmp.join("stock.tsv"), STOCK).unwrap();
    std::fs::write(tmp.join("fit.toml"), toml).unwrap();
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
    let mut args = vec!["fit", "predict", "--fit", "fit.toml", "--seed", "1", "--n-draws", "60"];
    args.extend_from_slice(extra);
    run(bin, tmp, &args)
}

// ── The headline: the free-forward first bin ────────────────────────────────

#[test]
fn free_forward_first_bin_opens_where_the_declaration_opens_it() {
    let bin = skip_if_missing_binary();
    let tmp = setup("ff", &fit_toml("results"), CASES);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let results = tmp.join("results");
    let cases_txt = std::fs::read_to_string(
        find_artifact(&results, "predictive", "cases").expect("predictive/cases.tsv"),
    )
    .unwrap();
    let cases = q50_by_time(&cases_txt, "free_forward");
    let (first, second) = (at(&cases, 150.0), at(&cases, 200.0));

    // The two declared bins are [100, 150) and [150, 200) — equal width, and
    // the flow rate is very nearly constant, so the medians must agree. Opening
    // the first bin at t = 0 makes it three times as wide, and its median ~3×.
    let ratio = first / second;
    assert!(
        (0.85..1.15).contains(&ratio),
        "the first incidence bin must open at its declared start, 100, giving \
         [100, 150) against [150, 200) — two 50-day bins of a near-constant \
         flow, so q50(150)/q50(200) ~ 1. Got {first} / {second} = {ratio:.3}; \
         a ratio near 3 means the bin opened at t_start = 0 and swallowed the \
         100-day warm-up.\n{cases_txt}"
    );

    // And the magnitude is right, not merely self-consistent: both bins sit
    // near the observed counts (4938, 4913), not near the 14888 the whole
    // [0, 150) window would give.
    for (t, v) in [(150.0, first), (200.0, second)] {
        assert!(
            (3500.0..6500.0).contains(&v),
            "predictive median at t = {t} is {v}; a 50-day bin of this model \
             carries ~4900 events.\n{cases_txt}"
        );
    }

    // ── The other direction: a PREVALENCE stream has no accumulator to reset,
    //    so no window touches it. `B(150)` is the whole-run cumulative stock —
    //    about three times one 50-day bin of flow.
    let stock_txt = std::fs::read_to_string(
        find_artifact(&results, "predictive", "stock").expect("predictive/stock.tsv"),
    )
    .unwrap();
    let stock = q50_by_time(&stock_txt, "free_forward");
    let stock_150 = at(&stock, 150.0);
    assert!(
        (12000.0..18000.0).contains(&stock_150),
        "the prevalence stream reads B at the instant t = 150 (~14888); got \
         {stock_150}.\n{stock_txt}"
    );
    assert!(
        stock_150 > 2.0 * first,
        "prevalence (a stock read from the origin, ~14888) and incidence (one \
         50-day bin, ~4938) must NOT coincide — if they do, the incidence bin \
         is still accumulating from t_start. stock={stock_150} cases={first}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The same property on the ODE backend (where gh#702 was found) ───────────

#[test]
fn free_forward_first_bin_is_backend_independent() {
    let bin = skip_if_missing_binary();
    let tmp = setup("ode", &ode_fit_toml("results"), CASES);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let cases_txt = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "predictive", "cases").expect("predictive/cases.tsv"),
    )
    .unwrap();
    let cases = q50_by_time(&cases_txt, "free_forward");
    let ratio = at(&cases, 150.0) / at(&cases, 200.0);
    assert!(
        (0.85..1.15).contains(&ratio),
        "a declared period is a property of the observation stream, not of the \
         forward backend: an ODE fit's first bin must open at 100 exactly as a \
         chain-binomial fit's does. Got ratio {ratio:.3}.\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── A period boundary off the recorded output grid is refused, never guessed ─

#[test]
fn a_period_boundary_off_the_output_grid_is_refused() {
    let bin = skip_if_missing_binary();
    // Under the 25-day declaration the row labelled 150 covers [125, 150).
    // t = 125 is on the dt = 1 grid the fit's filter walks, so the FIT is fine;
    // it is not a recorded output time (`every = 50 'days`), so the projection
    // has no cumulative flow to read there. Resolving it to the nearest earlier
    // snapshot (t = 100) would put 25 days of warm-up back into the first bin
    // and say nothing — exactly the silent-wrong gh#702 was about.
    let tmp = setup_with_model("offgrid", &fit_toml("results"), CASES_OFF_GRID, &model_off_grid());

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "a period boundary that is not a recorded output time must be refused, \
         not silently snapped to an earlier snapshot.\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("125") && stderr.contains("recorded output time"),
        "the refusal must name the boundary that is off the grid; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The one-step horizon: same window, a different mechanism ────────────────

#[test]
fn one_step_first_bin_opens_where_the_declaration_opens_it() {
    let bin = skip_if_missing_binary();
    let tmp = setup("os", &fit_toml("results"), CASES);

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "one_step"]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let cases_txt = std::fs::read_to_string(
        find_artifact(&tmp.join("results"), "predictive", "cases").expect("predictive/cases.tsv"),
    )
    .unwrap();
    let cases = q50_by_time(&cases_txt, "one_step");
    assert!(!cases.is_empty(), "a chain-binomial fit emits one_step rows:\n{cases_txt}");
    // The filter resets the bin where the declaration opens it (100), so the
    // one-step predictive at 150 is a 50-day bin, like the one at 200.
    let ratio = at(&cases, 150.0) / at(&cases, 200.0);
    assert!(
        (0.85..1.15).contains(&ratio),
        "the one-step first bin must open at the declared start: q50(150)/q50(200) \
         ~ 1, got {ratio:.3}.\n{cases_txt}"
    );
    // And no row is emitted AT the declared start: the filter stops there to
    // reset the bin, nothing is observed, nothing is plotted (gh#702).
    assert!(
        cases.iter().all(|(t, _)| (*t - 100.0).abs() > 1e-9),
        "no predictive row at the period's opening boundary t = 100:\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
