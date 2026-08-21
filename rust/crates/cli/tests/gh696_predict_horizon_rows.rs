//! gh#696 — `fit predict` emits the FREE-FORWARD stream bands out to the
//! model's own horizon, not only at the observation times.
//!
//! Within one `fit predict` run the trajectory is integrated to the model
//! horizon: `quantities/` carries rows out there, while `predictive/` used to
//! stop at the last observation. Same simulation, two artifacts, two time
//! axes — so a scenario's projected epidemic curve existed internally and was
//! discarded.
//!
//! What these tests pin:
//!
//!   * `free_forward_bands_run_to_the_model_horizon` — the headline: forecast
//!     rows appear past the last observation, on the stream's own reporting
//!     cadence, out to the horizon; they are not degenerate; the `one_step`
//!     band (data-conditioned by definition) still stops at the data; and the
//!     forecast agrees EXACTLY with what the same run's `quantities/` says
//!     about the same trajectory. The gh#561 scenario-horizon refusal is
//!     re-checked here, on the same fit, so this change cannot quietly make a
//!     declared-but-unhonourable horizon pass.
//!   * `horizon_equal_to_the_data_adds_no_rows` — the negative control: a model
//!     whose horizon is the last observation emits exactly the observed grid.
//!   * `survey_denominator_stream_gets_no_forecast_rows` — a likelihood whose
//!     trial denominator is an observed data column (`binomial(n = tested)`)
//!     has no denominator past the data. It is omitted from the extended grid
//!     with a named note, never extended with a zero denominator (which draws
//!     an identically-zero ribbon and reads as a forecast plateau).

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

/// The `time` values of every row in a predictive TSV whose `horizon` column
/// equals `horizon` and whose `scenario` column equals `scenario`, paired with
/// the row's `q50` cell (verbatim, so an exact-string oracle is possible).
fn rows_for<'a>(tsv: &'a str, scenario: &str, horizon: &str) -> Vec<(f64, &'a str, Vec<f64>)> {
    let mut lines = tsv.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let ix = |name: &str| header.iter().position(|h| *h == name).unwrap_or_else(|| panic!("column {name} in header {header:?}"));
    let (c_scen, c_time, c_hor, c_q50) = (ix("scenario"), ix("time"), ix("horizon"), ix("q50"));
    let (c_q05, c_q95) = (ix("q05"), ix("q95"));
    let mut out = Vec::new();
    for l in lines {
        let c: Vec<&str> = l.split('\t').collect();
        if c[c_scen] != scenario || c[c_hor] != horizon {
            continue;
        }
        let qs: Vec<f64> = [c_q05, c_q50, c_q95].iter().map(|&i| c[i].parse::<f64>().unwrap()).collect();
        out.push((c[c_time].parse::<f64>().unwrap(), c[c_q50], qs));
    }
    out
}

fn max_time(rows: &[(f64, &str, Vec<f64>)]) -> f64 {
    rows.iter().map(|(t, _, _)| *t).fold(f64::NEG_INFINITY, f64::max)
}

// ── The headline model: horizon 24 model-days past the last observation ─────

/// A closed SIR observed weekly to t = 56, with a model horizon of t = 80.
/// The forecast window is therefore 24 days wide and the stream's reporting
/// cadence is 7 days, so the free-forward band must gain rows at 63, 70, 77
/// (84 is past the horizon).
///
/// `cases_at_77` is the ORACLE: `value_at(observations.weekly_cases, 77 'days)`
/// reduces the very same per-draw `y_rep` series the predictive band is built
/// from. If the two files disagree at t = 77, one of them is wrong.
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
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  prevalence  = I / N
  cases_at_77 = value_at(observations.weekly_cases, 77 'days)
}

simulate {
  from = 0 'days
  to   = 80 'days
}

scenarios {
  longer { simulate { to = 120 'days } }
}
"#;

/// The same model with the horizon pulled back to the last observation — the
/// byte-level negative control (no forecast window ⇒ no forecast rows).
const MODEL_NO_HORIZON: &str = r#"
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
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 56 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml(model: &str) -> String {
    format!(
        r#"output_dir = "results"

[model]
camdl = "{model}"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 40
burn_in = 10
thin = 1
"#
    )
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_gh696_{}_{}_{}", tag, std::process::id(), ns));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn setup(tag: &str, model_src: &str) -> PathBuf {
    let tmp = tempdir(tag);
    std::fs::write(tmp.join("model.camdl"), model_src).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml("model.camdl")).unwrap();
    tmp
}

fn fit_then_predict(bin: &Path, tmp: &Path, predict_extra: &[&str]) -> std::process::Output {
    let out = run(bin, tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut args = vec!["fit", "predict", "--fit", "fit.toml"];
    args.extend_from_slice(predict_extra);
    run(bin, tmp, &args)
}

// ── The headline ────────────────────────────────────────────────────────────

#[test]
fn free_forward_bands_run_to_the_model_horizon() {
    let bin = skip_if_missing_binary();
    let tmp = setup("horizon", MODEL);

    let out = fit_then_predict(&bin, &tmp, &[]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let results = tmp.join("results");
    let pred = find_artifact(&results, "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();

    // ── The free-forward band reaches the model horizon on the stream's own
    //    weekly cadence: the observed grid 7…56, then 63, 70, 77.
    let ff = rows_for(&pred_txt, "fitted", "free_forward");
    let ff_times: Vec<f64> = ff.iter().map(|(t, _, _)| *t).collect();
    assert_eq!(
        ff_times,
        vec![7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0],
        "free_forward emits the observed times and then continues the stream's \
         7-day reporting cadence to the model horizon (t_end = 80; 84 is past \
         it). Got:\n{pred_txt}"
    );

    // ── The forecast rows carry a real band, not a fabricated flat ribbon.
    let forecast: Vec<&(f64, &str, Vec<f64>)> = ff.iter().filter(|(t, _, _)| *t > 56.0).collect();
    assert_eq!(forecast.len(), 3, "three forecast rows");
    for (t, _, qs) in &forecast {
        assert!(qs.iter().all(|q| q.is_finite()), "forecast band at t={t} must be finite: {qs:?}");
        for w in qs.windows(2) {
            assert!(w[0] <= w[1], "forecast quantiles ordered at t={t}: {qs:?}");
        }
    }
    assert!(
        forecast.iter().any(|(_, _, qs)| qs[2] > 0.0),
        "the forecast band must not be identically zero — a zero ribbon read as \
         a projection is the silent-wrong this feature must not produce. Got \
         {:?}",
        forecast.iter().map(|(t, _, qs)| (*t, qs.clone())).collect::<Vec<_>>()
    );

    // ── Negative control: the ONE-STEP band is data-conditioned by definition
    //    and must still stop at the last observation.
    let os = rows_for(&pred_txt, "fitted", "one_step");
    assert!(!os.is_empty(), "a chain-binomial fit emits one_step rows by default");
    assert_eq!(
        max_time(&os),
        56.0,
        "the one_step horizon conditions on data and must NOT gain forecast \
         rows. Got:\n{pred_txt}"
    );

    // ── The oracle: `quantities/` and `predictive/` come from ONE simulation.
    //    `cases_at_77 = value_at(observations.weekly_cases, 77 'days)` reduces
    //    the same per-draw y_rep series the band is built from, so the two
    //    files must agree at t = 77 to the byte.
    let q_at_77 = find_artifact(&results, "quantities", "cases_at_77")
        .expect("quantities/cases_at_77.tsv must be written");
    let q_txt = std::fs::read_to_string(&q_at_77).unwrap();
    let mut qlines = q_txt.lines();
    let qheader: Vec<&str> = qlines.next().expect("quantity header").split('\t').collect();
    let q50_col = qheader.iter().position(|h| *h == "q50").expect("q50 column");
    let qrow: Vec<&str> = qlines.next().expect("a cases_at_77 row").split('\t').collect();
    let band_q50 = ff
        .iter()
        .find(|(t, _, _)| *t == 77.0)
        .map(|(_, q50, _)| *q50)
        .expect("a free_forward row at t = 77");
    assert_eq!(
        qrow[q50_col], band_q50,
        "the forecast band and the quantity reduction read the SAME y_rep draws \
         off the SAME trajectory at t = 77 — they cannot disagree.\n\
         quantities/cases_at_77.tsv: {q_txt}\npredictive:\n{pred_txt}"
    );

    // ── And the trajectory really does reach the horizon (what made this
    //    change possible in the first place).
    let prev = find_artifact(&results, "quantities", "prevalence")
        .expect("quantities/prevalence.tsv must be written");
    let prev_txt = std::fs::read_to_string(&prev).unwrap();
    let pheader: Vec<&str> = prev_txt.lines().next().unwrap().split('\t').collect();
    let t_col = pheader.iter().position(|h| *h == "time").unwrap();
    let prev_max = prev_txt
        .lines()
        .skip(1)
        .map(|l| l.split('\t').nth(t_col).unwrap().parse::<f64>().unwrap())
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(prev_max, 80.0, "the quantity series runs to the model horizon");

    // ── gh#561 stays intact: a scenario declaring its OWN horizon is still
    //    refused, because `fit predict` runs the MODEL's window, not the
    //    scenario's. Extending to the model horizon must not open this door.
    let out = fit_then_predict(&bin, &tmp, &["--scenario", "longer"]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "scenario `longer` declares to = 120 against a model horizon of 80; \
         `fit predict` cannot honour it and must still refuse. stdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("declares a simulation horizon"),
        "the gh#561 refusal must name the declared horizon; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── Negative control: no forecast window ⇒ no forecast rows ─────────────────

#[test]
fn horizon_equal_to_the_data_adds_no_rows() {
    let bin = skip_if_missing_binary();
    let tmp = setup("nohorizon", MODEL_NO_HORIZON);

    let out = fit_then_predict(&bin, &tmp, &[]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let pred = find_artifact(&tmp.join("results"), "predictive", "weekly_cases")
        .expect("predictive/weekly_cases.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let ff = rows_for(&pred_txt, "fitted", "free_forward");
    let ff_times: Vec<f64> = ff.iter().map(|(t, _, _)| *t).collect();
    assert_eq!(
        ff_times,
        vec![7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0],
        "a model whose horizon IS the last observation gains nothing — the \
         emitted grid is exactly the observed one. Got:\n{pred_txt}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

// ── The data-supplied denominator ───────────────────────────────────────────

/// A survey-positivity stream whose binomial denominator is an observed data
/// column. Past the last observation there is no `tested` value, so the
/// denominator would resolve to 0 and every draw would be 0.
const MODEL_SURVEY: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate  in [0.05, 1.0] ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate  in [0.01, 0.5] ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
}

transitions {
  infection : S --> I  @ beta * S * I / N0
  recovery  : I --> R  @ gamma * I
}

init { S = 999  I = 1 }

observations {
  survey {
    columns       { time : time, pos : count, tested : count }
    projected     = prevalence(I)
    emit_schedule = every 2 'days
    pos ~ binomial(n = tested, p = projected / N0)
  }
}

quantities { infected = I }

simulate {
  from = 0 'days
  to   = 40 'days
}
"#;

const SURVEY_DATA: &str = "time\tpos\ttested\n2\t1\t100\n4\t2\t100\n6\t4\t100\n8\t7\t100\n\
10\t11\t100\n12\t16\t100\n14\t20\t100\n16\t24\t100\n18\t26\t100\n20\t25\t100\n";

#[test]
fn survey_denominator_stream_gets_no_forecast_rows() {
    let bin = skip_if_missing_binary();
    let tmp = tempdir("survey");
    std::fs::write(tmp.join("model.camdl"), MODEL_SURVEY).unwrap();
    std::fs::write(tmp.join("survey.tsv"), SURVEY_DATA).unwrap();
    std::fs::write(
        tmp.join("fit.toml"),
        r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
survey = "survey.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0 = 1000

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 40
burn_in = 10
thin = 1
"#,
    )
    .unwrap();

    let out = fit_then_predict(&bin, &tmp, &[]);
    assert!(
        out.status.success(),
        "fit predict failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    let pred = find_artifact(&tmp.join("results"), "predictive", "survey")
        .expect("predictive/survey.tsv must be written");
    let pred_txt = std::fs::read_to_string(&pred).unwrap();
    let ff = rows_for(&pred_txt, "fitted", "free_forward");

    // The model horizon is 40 and the data stops at 20, but the denominator is
    // exogenous: there is no `tested` past t = 20. The band must simply stop —
    // never continue as `binomial(0, p) = 0`.
    assert_eq!(
        max_time(&ff),
        20.0,
        "a stream whose denominator is an observed data column must NOT be \
         extended past the data. Got:\n{pred_txt}"
    );
    assert!(
        !ff.iter().any(|(t, _, _)| *t > 20.0),
        "no rows past the last observation for a data-denominator stream"
    );
    // And the omission is announced by name — silence here would look like the
    // model simply had no horizon.
    assert!(
        stderr.contains("survey") && stderr.contains("tested"),
        "the omission must name the stream and the data column that caused it; \
         stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
