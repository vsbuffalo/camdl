//! gh#702 — `fit predict` opens the first incidence bin at the fit's
//! `condition_from` boundary, not at the model origin.
//!
//! A `condition_from` fit simulates `[t_start, cond_from)` as warm-up and
//! scores its first incidence datum over `(cond_from, first_obs]`. The
//! predictive artifact is plotted against that same datum, so it must report
//! the flow over the SAME interval. It reported `(t_start, first_obs]` — the
//! whole warm-up folded into the first row — because the projection
//! (`project_all_obs_times`) seeds its cumulative-flow difference at zero and
//! the one-step filter never saw the conditioning hole the fit's likelihood
//! was given.
//!
//! Only the FIRST row is affected on the free-forward horizon: every later row
//! differences two observed times. On a single-observation fit the first row is
//! the only row, so the error is the whole artifact.
//!
//! ## The oracle
//!
//! The model is a one-way decay `A --> B @ mu * A` with `mu * t_end << 1`, so
//! the flow per unit time is very nearly constant. Observations sit at t = 150
//! and t = 200 with `condition_from = 100`, which makes the two scored bins
//! `(100, 150]` and `(150, 200]` — equal width, equal expected flow. The ratio
//! of the two predictive medians is therefore ~1 when the first bin opens at
//! the boundary, and ~3 when it opens at t = 0 (a 150-day bin against a 50-day
//! one). No fitted parameter enters the ratio, so the oracle does not depend on
//! how well the fit converged.
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
/// `cases` is an INTERVAL (incidence) stream — the flow accumulated since the
/// previous emitted time, which is what the conditioning boundary moves — and
/// `stock` is an INSTANT (prevalence) stream reading `B` itself, which has no
/// accumulator and must be untouched by the boundary.
///
/// `output { trajectories { every = 50 'days } }` puts the boundary t = 100 on
/// the recorded snapshot grid, which is what lets the projection read the
/// cumulative flow there at all.
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

/// Flow over (100, 150] and (150, 200] at mu = 1e-4, N0 = 1e6:
/// `N0 (e^{-mu t1} - e^{-mu t2})` = 4938 and 4913.
const CASES: &str = "time\tcases\n150\t4938\n200\t4913\n";

/// `B(t) = N0 (1 - e^{-mu t})` at t = 150 and 200 — the cumulative stock, which
/// no conditioning boundary resets.
const STOCK: &str = "time\tstock\n150\t14888\n200\t19801\n";

/// A chain-binomial fit, so BOTH predictive horizons are reachable
/// (`--horizon one_step` needs a filterable fit).
fn fit_toml(condition_from: &str, out_dir: &str) -> String {
    format!(
        r#"output_dir = "{out_dir}"
condition_from = "{condition_from}"

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
condition_from = "100"

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

fn setup(tag: &str, toml: &str) -> PathBuf {
    let tmp = tempdir(tag);
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("cases.tsv"), CASES).unwrap();
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
fn free_forward_first_bin_opens_at_the_conditioning_boundary() {
    let bin = skip_if_missing_binary();
    let tmp = setup("ff", &fit_toml("100", "results"));

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

    // The two scored bins are (100, 150] and (150, 200] — equal width, and the
    // flow rate is very nearly constant, so the medians must agree. Opening the
    // first bin at t = 0 makes it three times as wide, and its median ~3×.
    let ratio = first / second;
    assert!(
        (0.85..1.15).contains(&ratio),
        "the first incidence bin must open at condition_from = 100, giving \
         (100, 150] against (150, 200] — two 50-day bins of a near-constant \
         flow, so q50(150)/q50(200) ~ 1. Got {first} / {second} = {ratio:.3}; \
         a ratio near 3 means the bin opened at t_start = 0 and swallowed the \
         100-day warm-up.\n{cases_txt}"
    );

    // And the magnitude is right, not merely self-consistent: both bins sit
    // near the observed counts (4938, 4913), not near the 14888 the whole
    // (0, 150] window would give.
    for (t, v) in [(150.0, first), (200.0, second)] {
        assert!(
            (3500.0..6500.0).contains(&v),
            "predictive median at t = {t} is {v}; a 50-day bin of this model \
             carries ~4900 events.\n{cases_txt}"
        );
    }

    // ── The other direction: a PREVALENCE stream has no accumulator to reset,
    //    so the boundary must not touch it. `B(150)` is the whole-run
    //    cumulative stock — about three times one 50-day bin of flow.
    let stock_txt = std::fs::read_to_string(
        find_artifact(&results, "predictive", "stock").expect("predictive/stock.tsv"),
    )
    .unwrap();
    let stock = q50_by_time(&stock_txt, "free_forward");
    let stock_150 = at(&stock, 150.0);
    assert!(
        (12000.0..18000.0).contains(&stock_150),
        "the prevalence stream reads B at the instant t = 150 (~14888) and is \
         unaffected by the conditioning boundary; got {stock_150}.\n{stock_txt}"
    );
    assert!(
        stock_150 > 2.0 * first,
        "prevalence (a stock read from the origin, ~14888) and incidence (one \
         50-day bin, ~4938) must NOT coincide — if they do, the incidence bin \
         is still accumulating from t_start. stock={stock_150} cases={first}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The same defect on the ODE backend (where gh#702 was found) ─────────────

#[test]
fn free_forward_first_bin_is_backend_independent() {
    let bin = skip_if_missing_binary();
    let tmp = setup("ode", &ode_fit_toml("results"));

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
        "the conditioning boundary is a property of the observation window, \
         not of the forward backend: an ODE fit's first bin must open at \
         condition_from = 100 exactly as a chain-binomial fit's does. Got \
         ratio {ratio:.3}.\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The negative control: a boundary AT the origin still scores the lot ─────

#[test]
fn condition_from_at_the_origin_still_scores_the_whole_leading_window() {
    let bin = skip_if_missing_binary();
    // `condition_from = "0"` resolves to t_start, which is the documented
    // opt-in to scoring the FULL leading window (no warm-up discarded). The
    // first bin is then genuinely (0, 150] and must stay three times the
    // second — the seeding follows the RESOLVED boundary, and is not applied
    // just because the key is present.
    let tmp = setup("origin", &fit_toml("0", "results"));

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
        (2.5..3.6).contains(&ratio),
        "condition_from = 0 is the escape hatch: no warm-up is discarded, so \
         the first bin IS (0, 150] and stays ~3× the second. Got ratio \
         {ratio:.3} — the seeding must not fire for a boundary that resolved \
         to the origin.\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── A boundary off the recorded output grid is refused, never guessed ───────

#[test]
fn conditioning_boundary_off_the_output_grid_is_refused() {
    let bin = skip_if_missing_binary();
    // t = 125 is on the dt = 1 grid the fit's filter walks, so the FIT is fine;
    // it is not a recorded output time (`every = 50 'days`), so the projection
    // has no cumulative flow to read there. Resolving it to the nearest earlier
    // snapshot (t = 100) would put 25 days of warm-up back into the first bin
    // and say nothing — exactly the silent-wrong this issue is about.
    let tmp = setup("offgrid", &fit_toml("125", "results"));

    let out = fit_then_predict(&bin, &tmp, &["--horizon", "free_forward"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "a conditioning boundary that is not a recorded output time must be \
         refused, not silently snapped to an earlier snapshot.\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("condition_from") && stderr.contains("125"),
        "the refusal must name `condition_from` and the boundary that is off \
         the grid; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The one-step horizon: same window, a different mechanism ────────────────

#[test]
fn one_step_first_bin_opens_at_the_conditioning_boundary() {
    let bin = skip_if_missing_binary();
    let tmp = setup("os", &fit_toml("100", "results"));

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

    // The one-step band comes from the filter, not the projection: the filter
    // must be handed the SAME leading conditioning hole the fit's likelihood
    // was, or its first predictive is drawn from an accumulator that has been
    // running since t_start.
    let ratio = at(&cases, 150.0) / at(&cases, 200.0);
    assert!(
        (0.85..1.15).contains(&ratio),
        "the one_step band must condition on the same window the fit scored: \
         (100, 150] then (150, 200], two equal bins of a near-constant flow. \
         Got ratio {ratio:.3}; ~3 means the filter accumulated from \
         t_start.\n{cases_txt}"
    );

    // The conditioning hole is a reset, not an observation: it must not appear
    // as a predictive row at t = 100 (there is no observed counterpart for it,
    // and the row would be the discarded warm-up bin).
    assert!(
        !cases.iter().any(|(t, _)| (*t - 100.0).abs() < 1e-9),
        "no band row at the conditioning boundary t = 100 — it is a reset, not \
         an observation.\n{cases_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
