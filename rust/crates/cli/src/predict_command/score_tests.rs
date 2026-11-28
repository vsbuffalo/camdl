//! `camdl score` on the predictions of this module's fixture, checked by
//! running the release binary: the one-step score against `pfilter`'s
//! log-likelihood, the ODE score against the observation density computed
//! from `simulate`'s trajectory, the outcomes kept out of the prediction, the
//! refusal of an outcome file that disagrees with the prediction, the
//! recomputed prediction against the stored one, and the labels a score
//! carries.

use super::*;

/// `camdl score` in `cwd`, as a line of words.
fn score_in(cwd: &Path, root: &Path, line: &[&str]) -> Output {
    let args: Vec<String> = std::iter::once("score").chain(line.iter().copied()).map(str::to_string).collect();
    camdl(cwd, root, &args, &[])
}

/// Every stored score under `root`: its folder and identity record.
fn scores(root: &Path) -> Vec<(PathBuf, serde_json::Value)> {
    folders(root).into_iter().filter(|(_, id, _)| id["kind"] == crate::fit_store::Score::KIND).map(|(p, id, _)| (p, id)).collect()
}

/// The one score under `root`.
fn the_score(root: &Path) -> PathBuf {
    let found = scores(root);
    assert_eq!(found.len(), 1, "{found:#?}");
    found[0].0.clone()
}

fn read(folder: &Path, file: &str) -> String {
    std::fs::read_to_string(folder.join(file)).unwrap_or_else(|e| panic!("{}: {e}", folder.join(file).display()))
}

fn ok(out: &Output, what: &str) -> String {
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{what}:\n{err}");
    err
}

/// The seed `camdl predict` runs draw 0's filter at, for base seed `seed`
/// (`forecast_driver::draw_seed`), written out so a change to it fails here.
fn draw_zero_seed(seed: u64) -> u64 {
    seed ^ 0x9E37_79B9_7F4A_7C15
}

/// At a fixed point on the chain-binomial backend, with one copy of the
/// point, the per-row one-step log scores over every row sum to `pfilter`'s
/// log-likelihood at the same particles and seed: each row's score is the
/// log of the filter's incremental likelihood at that row (pomp's
/// `cond_logLik`), and they are computed from the same particles.
#[test]
fn the_one_step_score_sums_to_the_pfilter_log_likelihood() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
            "one_step", "--n-draws", "1", "--particles", "200", "--seed", "7", "--label", "os",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@os", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let folder = the_score(&root);
    let sum = |rows: Vec<BTreeMap<String, String>>| -> (usize, f64) {
        (rows.len(), rows.iter().map(|r| r["log_score"].parse::<f64>().unwrap()).sum())
    };
    let (n, per_stream) = sum(tsv_rows(&read(&folder, "scored_rows.tsv")));
    let (n_joint, joint) = sum(tsv_rows(&read(&folder, "joint_rows.tsv")));
    assert_eq!((n, n_joint), (63, 63), "every row of the one stream, and one joint row per row");

    let seed = draw_zero_seed(7).to_string();
    ok(
        &predict_in(cwd.path(), &root, &[
            "pfilter", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--particles",
            "200", "--seed", &seed,
        ]),
        "pfilter",
    );
    let pf = folders(&root).into_iter().find(|(_, id, _)| id["kind"] == "pfilter").expect("a stored pfilter run").0;
    let loglik: f64 = toml::from_str::<toml::Table>(&read(&pf, "loglik.toml")).unwrap()["loglik"].as_float().unwrap();
    assert!((per_stream - loglik).abs() <= 1e-9 * loglik.abs(), "per-stream sum {per_stream} vs pfilter {loglik}");
    assert!((joint - loglik).abs() <= 1e-9 * loglik.abs(), "joint sum {joint} vs pfilter {loglik}");
}

/// The natural log of `y!`, for a small count.
fn ln_factorial(y: u64) -> f64 {
    (2..=y).map(|k| (k as f64).ln()).sum()
}

/// On the ODE backend at a fixed point, the free-forward score of each row is
/// the observation density at the deterministic trajectory: here the Poisson
/// log mass of the observed count at `rho` times the day's infections, read
/// from `simulate`'s trajectory (four decimals) and computed here.
#[test]
fn the_ode_score_is_the_observation_density_at_the_trajectory() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--backend",
            "ode", "--n-draws", "1", "--label", "ode_point",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@ode_point", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let folder = the_score(&root);
    let manifest: serde_json::Value = serde_json::from_str(&read(&folder, "score.json")).unwrap();
    assert_eq!(manifest["modes"]["free_forward"]["basis"], serde_json::json!({"ode_free_forward": {"samples_per_draw": 1}}));
    let rows = tsv_rows(&read(&folder, "scored_rows.tsv"));

    let sim = predict_in(cwd.path(), &root, &["simulate", "model.camdl", "--params", "point.toml", "--backend", "ode", "-o", "traj.tsv"]);
    ok(&sim, "simulate");
    let traj = tsv_rows(&std::fs::read_to_string(cwd.path().join("traj.tsv")).unwrap());
    let flow_at = |t: f64| -> f64 {
        traj.iter().find(|r| r["time"].parse::<f64>().unwrap() == t).map(|r| r["flow_infection"].parse().unwrap()).unwrap()
    };
    let rho = 0.5;
    let mut n = 0;
    for r in rows.iter().filter(|r| r["stream"] == "daily_cases") {
        let t: f64 = r["t_stop"].parse().unwrap();
        let y: u64 = r["observed"].parse().unwrap();
        let rate = rho * flow_at(t);
        let expected = y as f64 * rate.ln() - rate - ln_factorial(y);
        let got: f64 = r["log_score"].parse().unwrap();
        assert!((got - expected).abs() < 1e-3, "t = {t}: score {got} vs density {expected}");
        n += 1;
    }
    assert_eq!(n, 63);
}

/// The outcomes enter the score and never the prediction: an edited held-out
/// outcome makes another score of the same prediction, and the prediction's
/// files are the same bytes before and after it is scored.
#[test]
fn a_held_out_outcome_changes_the_score_and_never_the_prediction() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
            "one_step", "--n-draws", "2", "--particles", "40", "--holdout-after", "20", "--label", "held",
        ]),
        "predict",
    );
    let found = predictions(&root);
    assert_eq!(found.len(), 1);
    let (prediction, before_id) = found[0].clone();
    let before = files_under(&prediction);
    ok(&score_in(cwd.path(), &root, &["@held", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let edited: String = daily
        .lines()
        .map(|l| match l.split_once('\t') {
            Some(("40", _)) => "40\t99\n".to_string(),
            _ => format!("{l}\n"),
        })
        .collect();
    assert_ne!(edited, daily);
    std::fs::write(cwd.path().join("edited.tsv"), &edited).unwrap();
    ok(&score_in(cwd.path(), &root, &["@held", "--data", "daily_cases=edited.tsv"]), "score of the edited outcome");
    let ids: Vec<String> = scores(&root).iter().map(|(_, id)| id["run_id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids.len(), 2, "the edited outcome is another score: {ids:?}");
    assert_ne!(ids[0], ids[1]);
    let after = predictions(&root);
    assert_eq!(after.len(), 1, "no prediction is added");
    assert_eq!(after[0].1, before_id, "the prediction's identity is unchanged");
    let mut now = files_under(&prediction);
    now.retain(|name, _| !name.starts_with("scores/"));
    assert_eq!(now, before, "the prediction's files are the same bytes with and without its scores");
    // The rows before the edited one score identically: the pass that
    // predicts them has assimilated nothing of it.
    let by_time = |folder: &Path| -> BTreeMap<String, String> {
        tsv_rows(&read(folder, "scored_rows.tsv"))
            .into_iter()
            .filter(|r| r["stream"] == "daily_cases")
            .map(|r| (r["t_stop"].clone(), r["log_score"].clone()))
            .collect()
    };
    let found = scores(&root);
    let (a, b) = (by_time(&found[0].0), by_time(&found[1].0));
    let mut differ: Vec<f64> = a.iter().filter(|(t, s)| b.get(*t) != Some(s)).map(|(t, _)| t.parse().unwrap()).collect();
    differ.sort_by(f64::total_cmp);
    assert_eq!(differ.first(), Some(&40.0), "the first row whose score moves is the edited one: {differ:?}");
}

/// An outcome file whose rows differ from the ones the prediction conditioned
/// on, in one row, is refused, naming the stream and the row.
#[test]
fn an_outcome_file_that_disagrees_with_the_prediction_is_refused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
            "free_forward", "--n-draws", "2", "--holdout-after", "20", "--label", "ff",
        ]),
        "predict",
    );
    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let edited: String = daily
        .lines()
        .map(|l| match l.split_once('\t') {
            Some(("10", _)) => "10\t77\n".to_string(),
            _ => format!("{l}\n"),
        })
        .collect();
    std::fs::write(cwd.path().join("edited.tsv"), edited).unwrap();
    let out = score_in(cwd.path(), &root, &["@ff", "--data", "daily_cases=edited.tsv"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("row 10 of 'daily_cases'") && err.contains("77"), "names the stream, the row and its value: {err}");
    assert!(scores(&root).is_empty(), "nothing is stored");
    // A file that lacks a conditioning row is refused too.
    let short: String = daily.lines().take(5).map(|l| format!("{l}\n")).collect();
    std::fs::write(cwd.path().join("short.tsv"), short).unwrap();
    let out = score_in(cwd.path(), &root, &["@ff", "--data", "daily_cases=short.tsv"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("row 5 (time 5) is missing"), "{err}");
}

/// Every mode each backend has, from both parameter sources, is scored: the
/// score computes the prediction again with its outcomes bound, and writes a
/// score only when the run id and every file it computes equal the stored
/// prediction's byte for byte. A free-forward or one-step score's
/// `summary_rows.tsv` then has one row per (stream, mode, role), and a
/// forecast's `summary_targets.tsv` one per (stream, role, lead), each the
/// count of the scored rows with that key.
#[test]
fn the_score_recomputes_every_prediction_and_pools_nothing() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let data = ["--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv", "--data", "survey=survey.tsv"];
    let lines: Vec<(&str, Vec<&str>, Vec<&str>)> = vec![
        ("posterior, both modes", vec!["predict", "@fit", "--n-draws", "6", "--particles", "20"], data.to_vec()),
        ("posterior forecast", vec!["predict", "@fit", "--horizon", "2 weeks", "--stream", "daily_cases", "--stream", "weekly_cases", "--n-draws", "6", "--particles", "20"], data.to_vec()),
        ("ODE posterior", vec!["predict", "@ode", "--n-draws", "4"], data.to_vec()),
        ("fixed point, both modes", vec!["predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--n-draws", "3", "--particles", "20", "--holdout-after", "24"], vec!["--data", "daily_cases=daily_cases.tsv"]),
        ("fixed-point forecast", vec!["predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--horizon", "10 days", "--n-draws", "3", "--particles", "20", "--holdout-after", "24"], vec!["--data", "daily_cases=daily_cases.tsv"]),
    ];
    for (i, (what, predict, outcomes)) in lines.iter().enumerate() {
        let label = format!("p{i}");
        let line = [predict.as_slice(), &["--label", &label]].concat();
        ok(&predict_in(cwd.path(), &root, &line), what);
        let handle = format!("@{label}");
        let err = ok(&score_in(cwd.path(), &root, &[[handle.as_str()].as_slice(), outcomes.as_slice()].concat()), what);
        assert!(err.contains("row(s) scored"), "{what}: {err}");
    }
    let found = scores(&root);
    assert_eq!(found.len(), lines.len(), "{found:#?}");
    for (folder, _) in &found {
        let forecast = folder.to_string_lossy().contains("/forecast-");
        let (rows_file, summary_file, by) = match forecast {
            true => ("scored_targets.tsv", "summary_targets.tsv", "lead"),
            false => ("scored_rows.tsv", "summary_rows.tsv", "mode"),
        };
        let rows = tsv_rows(&read(folder, rows_file));
        let summary = tsv_rows(&read(folder, summary_file));
        assert!(!rows.is_empty() && !summary.is_empty(), "{}", folder.display());
        let key = |r: &BTreeMap<String, String>| (r["scenario"].clone(), r["stream"].clone(), r["role"].clone(), r[by].clone());
        let mut counted: BTreeMap<_, usize> = BTreeMap::new();
        for r in &rows {
            *counted.entry(key(r)).or_default() += 1;
        }
        let keys: Vec<_> = summary.iter().map(key).collect();
        let distinct: std::collections::BTreeSet<_> = keys.iter().cloned().collect();
        assert_eq!(keys.len(), distinct.len(), "one summary row per key: {}", folder.display());
        for s in &summary {
            assert_eq!(s["n"].parse::<usize>().unwrap(), counted[&key(s)], "{s:?}");
        }
        assert_eq!(counted.len(), summary.len(), "every key of the scored rows is summarized");
    }
    // The posterior's scores keep its trained and held-out rows apart, and
    // a forecast's targets apart by lead.
    let all = tsv_rows(&read(&found.iter().find(|(p, _)| p.to_string_lossy().contains("/predict/all-")).unwrap().0, "summary_rows.tsv"));
    assert!(all.iter().any(|r| r["role"] == "trained") && all.iter().any(|r| r["role"] == "held_out"), "{all:?}");
    let fc = tsv_rows(&read(&found.iter().find(|(p, _)| p.to_string_lossy().contains("/predict/forecast-")).unwrap().0, "summary_targets.tsv"));
    let leads: std::collections::BTreeSet<&str> = fc.iter().filter(|r| r["stream"] == "daily_cases").map(|r| r["lead"].as_str()).collect();
    assert!(leads.len() > 1, "one summary row per lead: {fc:?}");
}

/// A fixed point's rows after `--holdout-after` are held out as declared,
/// which the score records and labels as not verified; without
/// `--holdout-after` its conditioning is unknown.
#[test]
fn a_fixed_points_held_out_rows_are_labelled_as_declared() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let base = ["predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode", "one_step", "--n-draws", "2", "--particles", "30"];
    ok(&predict_in(cwd.path(), &root, &[base.as_slice(), &["--holdout-after", "20", "--label", "declared"]].concat()), "predict");
    ok(&score_in(cwd.path(), &root, &["@declared", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let folder = the_score(&root);
    let manifest: serde_json::Value = serde_json::from_str(&read(&folder, "score.json")).unwrap();
    assert_eq!(manifest["conditioning"], serde_json::json!({"hold_out_declared": {"trained_through": 20.0}}), "{manifest:#}");
    assert!(manifest["labels"].as_array().unwrap().contains(&serde_json::json!("held out as declared, not verified")), "{manifest:#}");
    let rows = tsv_rows(&read(&folder, "scored_rows.tsv"));
    let role_at = |t: f64| rows.iter().find(|r| r["stream"] == "daily_cases" && r["t_stop"].parse::<f64>().unwrap() == t).unwrap()["role"].clone();
    assert_eq!(role_at(20.0), "trained");
    assert_eq!(role_at(21.0), "held_out");

    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(&predict_in(cwd.path(), &root, &[base.as_slice(), &["--label", "unknown"]].concat()), "predict");
    ok(&score_in(cwd.path(), &root, &["@unknown", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let folder = the_score(&root);
    let manifest: serde_json::Value = serde_json::from_str(&read(&folder, "score.json")).unwrap();
    assert_eq!(manifest["conditioning"], serde_json::json!("unknown"), "{manifest:#}");
    assert!(tsv_rows(&read(&folder, "scored_rows.tsv")).iter().all(|r| r["role"] == "unknown"));
}

/// A forecast read under `--covariate pop=realised` is labelled conditional
/// on covariates observed after the origin, and so is its score.
#[test]
fn a_realised_covariate_labels_the_score() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let line = [&["predict"][..], &forced_forecast(&["--covariate", "pop=realised", "--label", "realised"])].concat();
    ok(&predict_in(cwd.path(), &root, &line), "predict");
    ok(&score_in(cwd.path(), &root, &["@realised", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let folder = the_score(&root);
    let manifest: serde_json::Value = serde_json::from_str(&read(&folder, "score.json")).unwrap();
    assert_eq!(manifest["covariates"], serde_json::json!({"pop": "realised"}), "{manifest:#}");
    assert!(
        manifest["labels"].as_array().unwrap().contains(&serde_json::json!("conditional on covariates observed after the origin")),
        "{manifest:#}"
    );
    // The same forecast stated prospective carries no such label.
    let line = [&["predict"][..], &forced_forecast(&["--covariate", "pop=prospective", "--label", "prospective"])].concat();
    ok(&predict_in(cwd.path(), &root, &line), "predict");
    ok(&score_in(cwd.path(), &root, &["@prospective", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let other = scores(&root).into_iter().find(|(p, _)| *p != folder).unwrap().0;
    let manifest: serde_json::Value = serde_json::from_str(&read(&other, "score.json")).unwrap();
    assert!(manifest["labels"].as_array().unwrap().is_empty(), "{manifest:#}");
}

/// `camdl show` and `camdl cat` read a score: `show` names the prediction it
/// scores, and `cat` prints its summary by default.
#[test]
fn show_and_cat_read_a_score() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(&predict_in(cwd.path(), &root, &[FIXED, &["--label", "ff"]].concat()), "predict");
    ok(&score_in(cwd.path(), &root, &["@ff", "--data", "daily_cases=daily_cases.tsv", "--label", "sc"]), "score");
    let folder = the_score(&root);
    let show = camdl(cwd.path(), &root, &["show".into(), "@sc".into()], &[]);
    let text = String::from_utf8_lossy(&show.stdout);
    assert!(show.status.success(), "{}", String::from_utf8_lossy(&show.stderr));
    assert!(text.contains("score") && text.contains("summary_rows.tsv"), "{text}");
    let cat = camdl(cwd.path(), &root, &["cat".into(), folder.to_string_lossy().into_owned()], &[]);
    assert!(cat.status.success(), "{}", String::from_utf8_lossy(&cat.stderr));
    assert_eq!(String::from_utf8_lossy(&cat.stdout), read(&folder, "summary_rows.tsv"));
}

/// Two observation blocks that read one data source through `from surv`.
const SHARED: &str = r#"
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta  : rate        in [0.001, 2.0]
  gamma : rate        in [0.001, 1.0]
  rho   : probability in [0.0,   1.0]
  N0    : count       in [100,   100000]
  I0    : count       in [1,     1000]
  k     : real        in [0.1,   100.0]
}
transitions {
  infection : S --> I  @ beta * S * (I / N)
  recovery  : I --> R  @ gamma * I
}
observations {
  weekly from surv {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ neg_binomial(mean = rho * projected, r = k)
  }
  recovered from surv {
    columns       { time : time, recs : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(recovery)
    emit_schedule = every 7 'days
    recs          ~ neg_binomial(mean = rho * projected, r = k)
  }
}
init {
  S = N0 - I0
  I = I0
}
simulate { from = 0 'days  to = 63 'days }
"#;

/// Two blocks that read one source are two streams: each keeps its own name
/// in `scored_rows.tsv`, beside the source, and its own rows in
/// `summary_rows.tsv`. A
/// fixed point bound through the blocks of a shared source is scored.
#[test]
fn two_streams_reading_one_source_are_scored_apart() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let surv = (1..=8).fold("time\tcases\trecs\n".to_string(), |s, w| s + &format!("{}\t{}\t{}\n", 7 * w, 3 + 2 * w, w));
    std::fs::write(cwd.path().join("shared.camdl"), SHARED).unwrap();
    std::fs::write(cwd.path().join("surv.tsv"), surv).unwrap();
    std::fs::write(cwd.path().join("shared.toml"), "beta = 0.5\ngamma = 0.25\nrho = 0.5\nN0 = 1000\nI0 = 5\nk = 5.0\n").unwrap();
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "shared.camdl", "--params", "shared.toml", "--data", "weekly=surv.tsv", "--data", "recovered=surv.tsv",
            "--mode", "one_step", "--n-draws", "2", "--particles", "30", "--holdout-after", "28", "--label", "shared",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@shared", "--data", "surv=surv.tsv"]), "score");
    let folder = the_score(&root);
    let rows = tsv_rows(&read(&folder, "scored_rows.tsv"));
    for stream in ["weekly", "recovered"] {
        let mine: Vec<_> = rows.iter().filter(|r| r["stream"] == stream).collect();
        assert_eq!(mine.len(), 8, "{stream}: {rows:?}");
        assert!(mine.iter().all(|r| r["source"] == "surv"));
    }
    let summary = tsv_rows(&read(&folder, "summary_rows.tsv"));
    for stream in ["weekly", "recovered"] {
        for (role, n) in [("trained", "4"), ("held_out", "4")] {
            let r: Vec<_> = summary.iter().filter(|r| r["stream"] == stream && r["role"] == role).collect();
            assert_eq!(r.len(), 1, "{stream} {role}: {summary:?}");
            assert_eq!(r[0]["n"], n, "never pooled across the blocks of one source: {summary:?}");
        }
    }
}

/// A fit with no holdout scores in sample on its own rows, and the outcome
/// rows past its data are held out, so a score that reads them is labelled
/// held out from the end of the fit's last row.
#[test]
fn a_fit_with_no_holdout_is_labelled_held_out_past_its_data() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let cut: String = daily.lines().take(25).map(|l| format!("{l}\n")).collect();
    std::fs::write(cwd.path().join("daily24.tsv"), cut).unwrap();
    let fit = super::FIT.replace("holdout_after = 24.0\n", "").replace(
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"\nsurvey = \"survey.tsv\"",
        "daily_cases = \"daily24.tsv\"",
    );
    assert!(!fit.contains("holdout_after") && fit.contains("daily24.tsv"), "the fixture's fit moved");
    std::fs::write(cwd.path().join("fit_all.toml"), fit).unwrap();
    ok(&predict_in(cwd.path(), &root, &["fit", "run", "fit_all.toml", "--label", "all"]), "fit");
    ok(
        &predict_in(cwd.path(), &root, &["predict", "@all", "--mode", "one_step", "--n-draws", "3", "--particles", "20", "--label", "os_all"]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@os_all", "--data", "daily_cases=daily24.tsv"]), "score on its own rows");
    ok(&score_in(cwd.path(), &root, &["@os_all", "--data", "daily_cases=daily_cases.tsv"]), "score past its rows");
    let found = scores(&root);
    assert_eq!(found.len(), 2);
    let mut seen = Vec::new();
    for (folder, _) in &found {
        let m: serde_json::Value = serde_json::from_str(&read(folder, "score.json")).unwrap();
        let roles: Vec<String> = tsv_rows(&read(folder, "scored_rows.tsv")).into_iter().map(|r| r["role"].clone()).collect();
        seen.push((m["conditioning"].clone(), roles.iter().any(|r| r == "held_out")));
    }
    assert!(seen.contains(&(serde_json::json!("in_sample"), false)), "{seen:?}");
    assert!(seen.iter().any(|(c, held)| *held && c["hold_out_tail"]["train_end"] == serde_json::json!(24.0)), "{seen:?}");
}

/// Under `--holdout-after`, a fixed point's one-step score of the whole file
/// sums over every row to `pfilter`'s log-likelihood of the whole file: the
/// pass through the held-out rows is the filter run over them.
#[test]
fn the_onward_pass_ties_with_the_pfilter_on_the_whole_file() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
            "one_step", "--n-draws", "1", "--particles", "200", "--seed", "7", "--holdout-after", "20", "--label", "held",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@held", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let rows = tsv_rows(&read(&the_score(&root), "scored_rows.tsv"));
    let rows: Vec<_> = rows.iter().filter(|r| r["stream"] == "daily_cases").collect();
    assert_eq!(rows.len(), 63);
    assert!(rows.iter().any(|r| r["role"] == "held_out"));
    let sum: f64 = rows.iter().map(|r| r["log_score"].parse::<f64>().unwrap()).sum();
    let seed = draw_zero_seed(7).to_string();
    ok(
        &predict_in(cwd.path(), &root, &[
            "pfilter", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--particles",
            "200", "--seed", &seed,
        ]),
        "pfilter",
    );
    let pf = folders(&root).into_iter().find(|(_, id, _)| id["kind"] == "pfilter").unwrap().0;
    let loglik: f64 = toml::from_str::<toml::Table>(&read(&pf, "loglik.toml")).unwrap()["loglik"].as_float().unwrap();
    assert!((sum - loglik).abs() <= 1e-9 * loglik.abs(), "score sum {sum} vs pfilter {loglik}");
}

/// The one-step prediction at `--seed SEED` of two copies of the point at 3
/// particles, the rows after day 20 held out, and the store and working
/// directory it ran in.
fn degenerating(seed: &str) -> (tempfile::TempDir, PathBuf, tempfile::TempDir, Output) {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let out = predict_in(cwd.path(), &root, &[
        "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
        "one_step", "--n-draws", "2", "--particles", "3", "--seed", seed, "--holdout-after", "20", "--label", "deg",
    ]);
    (fresh, root, cwd, out)
}

/// A draw whose one-step filter collapses on a held-out row fails the score,
/// as it fails a forecast: dropping it would leave out the draw under which
/// the outcomes were least probable. At seed 4, draw 0 of two collapses at
/// t = 29, after the conditioning rows, so the prediction succeeds and the
/// score fails, naming the draw and the remedy.
#[test]
fn a_draw_collapsing_on_a_held_out_row_fails_the_score() {
    let (_fresh, root, cwd, out) = degenerating("4");
    ok(&out, "predict");
    let o = score_in(cwd.path(), &root, &["@deg", "--data", "daily_cases=daily_cases.tsv"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "a collapsed draw must fail the score:\n{err}");
    assert!(err.contains("1 of 2 posterior draws collapsed") && err.contains("draw 0"), "{err}");
    assert!(err.contains("--particles"), "{err}");
}

/// A draw whose one-step filter collapses on a conditioning row fails the
/// prediction. At seed 2, draw 1 of two collapses before day 20.
#[test]
fn a_draw_collapsing_on_a_conditioning_row_fails_the_prediction() {
    let (_fresh, _root, _cwd, out) = degenerating("2");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a collapsed draw must fail the prediction:\n{err}");
    assert!(err.contains("1 of 2 posterior draws collapsed") && err.contains("draw 1"), "{err}");
}

/// The refusals of `--data` and of the outcome rows.
#[test]
fn the_outcome_refusals() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--horizon",
            "5 days", "--n-draws", "2", "--particles", "10", "--holdout-after", "20", "--label", "fc",
        ]),
        "predict",
    );
    let refused = |line: &[&str], says: &str| {
        let out = score_in(cwd.path(), &root, line);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{line:?}: {err}");
        assert!(err.contains(says), "{line:?}: {err}");
    };
    refused(&["@fc", "--data", "survey=survey.tsv"], "binds no stream source `survey`");
    refused(&["@fc", "--data", "daily_cases=daily_cases.tsv", "--data", "daily_cases=daily_cases.tsv"], "names stream `daily_cases` twice");
    refused(&["@fc", "--data", "daily_cases=absent.tsv"], "absent.tsv: it does not exist");
    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let dup: String =
        daily.lines().flat_map(|l| if l.starts_with("22\t") { vec![l, l] } else { vec![l] }).map(|l| format!("{l}\n")).collect();
    std::fs::write(cwd.path().join("dup.tsv"), dup).unwrap();
    refused(&["@fc", "--data", "daily_cases=dup.tsv"], "both measure the target ending at 22");
    assert!(scores(&root).is_empty(), "nothing is stored");
}

/// A stored prediction file edited by hand is refused before anything is
/// scored: the store finds the file's digest changed, with its size and
/// modification time kept.
#[test]
fn a_hand_edited_prediction_is_refused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(&predict_in(cwd.path(), &root, &[FIXED, &["--label", "ff"]].concat()), "predict");
    let (folder, _) = predictions(&root).into_iter().next().unwrap();
    let file = folder.join("predictive/daily_cases.tsv");
    let mut bytes = std::fs::read(&file).unwrap();
    let last = bytes.len() - 2;
    bytes[last] = if bytes[last] == b'1' { b'2' } else { b'1' };
    let mtime = std::fs::metadata(&file).unwrap().modified().unwrap();
    let mut perms = std::fs::metadata(&file).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    std::fs::set_permissions(&file, perms).unwrap();
    std::fs::write(&file, bytes).unwrap();
    std::fs::File::options().write(true).open(&file).unwrap().set_modified(mtime).unwrap();
    let out = score_in(cwd.path(), &root, &["@ff", "--data", "daily_cases=daily_cases.tsv"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("damaged") && err.contains("predictive/daily_cases.tsv"), "{err}");
    assert!(scores(&root).is_empty());
}

/// An outcome row after the conditioning rows that ends before the last of
/// them is refused: the one-step pass would assimilate it before rows the
/// prediction predicted. The data loader refuses such a file, whose rows are
/// out of time order, before the pass is posed; a stored fit and a fixed
/// point's `--holdout-after` both cut by a row's end, so their held-out rows
/// all end after their conditioning rows.
#[test]
fn an_outcome_row_before_the_last_conditioning_row_is_refused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
            "one_step", "--n-draws", "2", "--particles", "10", "--holdout-after", "20", "--label", "os",
        ]),
        "predict",
    );
    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let early: String = daily
        .lines()
        .flat_map(|l| if l.starts_with("20\t") { vec![l.to_string(), "15\t3".to_string()] } else { vec![l.to_string()] })
        .map(|l| l + "\n")
        .collect();
    std::fs::write(cwd.path().join("early.tsv"), early).unwrap();
    let out = score_in(cwd.path(), &root, &["@os", "--data", "daily_cases=early.tsv"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("not in chronological order"), "{err}");
    assert!(scores(&root).is_empty());
}

/// The fixture model with `daily_cases` replaced by a stream of calendar
/// months, each row's period given by its window columns, run to day 120,
/// and its data file: four months of 31, 28, 31 and 30 days.
fn monthly(cwd: &Path) {
    let model = std::fs::read_to_string(cwd.join("model.camdl")).unwrap();
    let monthly = model
        .replace(
            "  daily_cases {\n    columns       { time : time, daily_cases : count }\n    covers        = closing_at(time, 1 'days)\n    projected     = incidence(infection)\n    emit_schedule = every 1 'days\n",
            "  daily_cases {\n    columns       { start : window_start, stop : window_stop, daily_cases : count }\n    projected     = incidence(infection)\n",
        )
        .replace("to = 63 'days", "to = 120 'days");
    assert!(monthly.contains("window_start") && monthly.contains("to = 120 'days"), "the fixture's daily stream moved");
    std::fs::write(cwd.join("monthly.camdl"), monthly).unwrap();
    std::fs::write(cwd.join("months.tsv"), "start\tstop\tdaily_cases\n0\t31\t385\n31\t59\t6\n59\t90\t0\n90\t120\t0\n").unwrap();
}

/// A one-step score of a stream of calendar months has one summary row per
/// role: the rows' periods of 28 to 31 days do not split the stream's
/// summary.
#[test]
fn a_one_step_score_of_a_monthly_stream_has_one_summary_row_per_role() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    monthly(cwd.path());
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "monthly.camdl", "--params", "point.toml", "--data", "daily_cases=months.tsv", "--mode",
            "one_step", "--n-draws", "2", "--particles", "100", "--holdout-after", "59", "--label", "months",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@months", "--data", "daily_cases=months.tsv"]), "score");
    let summary = tsv_rows(&read(&the_score(&root), "summary_rows.tsv"));
    let mine: Vec<_> = summary.iter().filter(|r| r["stream"] == "daily_cases").collect();
    let roles: Vec<(&str, &str)> = mine.iter().map(|r| (r["role"].as_str(), r["n"].as_str())).collect();
    assert_eq!(roles, [("trained", "2"), ("held_out", "2")], "one summary row per role: {summary:?}");
}

/// The header row of a score's file `file`.
fn header_of(folder: &Path, file: &str) -> Vec<String> {
    read(folder, file).lines().next().unwrap_or_default().split('\t').map(str::to_string).collect()
}

/// A one-step score of the daily and the weekly stream at a fixed point, the
/// rows after day 24 held out: its folder.
fn two_stream_one_step(root: &Path, cwd: &Path) -> PathBuf {
    let data = ["--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv"];
    ok(
        &predict_in(cwd, root, &[
            &["predict", "model.camdl", "--params", "point.toml"][..], &data,
            &["--mode", "one_step", "--n-draws", "2", "--particles", "40", "--holdout-after", "24", "--label", "two"],
        ]
        .concat()),
        "predict",
    );
    ok(&score_in(cwd, root, &[&["@two"][..], &data].concat()), "score");
    the_score(root)
}

/// The joint rows of a one-step score are in files of their own: no
/// per-stream file holds one, and the joint summary's held-out
/// `log_score_sum` is the sum of the held-out joint rows' log scores, the
/// one-step score of every held-out row together.
#[test]
fn a_one_step_scores_joint_rows_are_in_their_own_files() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let folder = two_stream_one_step(&root, cwd.path());
    for file in ["scored_rows.tsv", "summary_rows.tsv"] {
        let rows = tsv_rows(&read(&folder, file));
        assert!(!rows.is_empty(), "{file}");
        assert!(rows.iter().all(|r| r["stream"] == "daily_cases" || r["stream"] == "weekly_cases"), "{file}: {rows:?}");
        assert!(!header_of(&folder, file).contains(&"streams".to_string()), "{file}");
    }
    let joints = tsv_rows(&read(&folder, "joint_rows.tsv"));
    let held: Vec<f64> = joints.iter().filter(|r| r["role"] == "held_out").map(|r| r["log_score"].parse().unwrap()).collect();
    assert!(!held.is_empty(), "{joints:?}");
    assert!(
        joints.iter().any(|r| r["streams"] == "daily_cases,weekly_cases"),
        "a stop where both streams end scores them together: {joints:?}"
    );
    let summary = tsv_rows(&read(&folder, "joint_summary_rows.tsv"));
    let row: Vec<_> = summary.iter().filter(|r| r["role"] == "held_out").collect();
    assert_eq!(row.len(), 1, "{summary:?}");
    assert_eq!(row[0]["n"], held.len().to_string());
    let sum: f64 = row[0]["log_score_sum"].parse().unwrap();
    let want: f64 = held.iter().sum();
    assert!((sum - want).abs() <= 1e-9 * want.abs().max(1.0), "joint summary {sum} vs the joint rows' sum {want}");
}

/// The two layouts share their first columns through `in_90`. A forecast
/// score's files carry `origin`, `periods_ahead` and `lead` and no `mode`;
/// a data-row score's carry `mode` and none of the three.
#[test]
fn the_two_layouts_share_a_core_and_carry_their_own_columns() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let rows = two_stream_one_step(&root, cwd.path());
    ok(
        &predict_in(cwd.path(), &root, &[
            "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--horizon",
            "10 days", "--n-draws", "2", "--particles", "20", "--holdout-after", "24", "--label", "fc",
        ]),
        "predict",
    );
    ok(&score_in(cwd.path(), &root, &["@fc", "--data", "daily_cases=daily_cases.tsv"]), "score");
    let targets = scores(&root).into_iter().map(|(p, _)| p).find(|p| *p != rows).unwrap();
    let has = |folder: &Path, file: &str, col: &str| header_of(folder, file).iter().any(|c| c == col);
    for file in ["scored_rows.tsv", "summary_rows.tsv", "joint_rows.tsv", "joint_summary_rows.tsv"] {
        for col in ["origin", "periods_ahead", "lead"] {
            assert!(!has(&rows, file, col), "{file} has no {col}");
        }
        assert!(!rows.join(file.replace("rows", "targets")).exists(), "a data-row score writes no target file");
    }
    for file in ["scored_rows.tsv", "summary_rows.tsv"] {
        assert!(has(&rows, file, "mode"), "{file} has a mode");
    }
    for file in ["scored_targets.tsv", "summary_targets.tsv", "joint_targets.tsv", "joint_summary_targets.tsv"] {
        assert!(!has(&targets, file, "mode"), "{file} has no mode");
        assert!(has(&targets, file, "lead"), "{file} has a lead");
        assert!(!targets.join(file.replace("targets", "rows")).exists(), "a forecast score writes no data-row file");
    }
    for col in ["origin", "periods_ahead"] {
        assert!(has(&targets, "scored_targets.tsv", col), "scored_targets.tsv has {col}");
    }
    let core = |folder: &Path, file: &str| -> Vec<String> {
        let h = header_of(folder, file);
        let end = h.iter().position(|c| c == "in_90").expect("the core ends at in_90") + 1;
        h[..end].to_vec()
    };
    assert_eq!(core(&rows, "scored_rows.tsv"), core(&targets, "scored_targets.tsv"), "one shared core");
    let manifest: serde_json::Value = serde_json::from_str(&read(&targets, "score.json")).unwrap();
    assert_eq!(manifest["schema"], "camdl.score/v2");
    assert!(manifest["modes"]["forecast"].is_object(), "score.json records the mode: {manifest:#}");
}

/// Each row of `stream` in a score's `file`, as `(t_stop, assimilated_through)`.
fn assimilated(folder: &Path, file: &str, stream: Option<&str>) -> Vec<(String, String)> {
    tsv_rows(&read(folder, file))
        .into_iter()
        .filter(|r| stream.is_none_or(|s| r["stream"] == s))
        .map(|r| (r["t_stop"].clone(), r["assimilated_through"].clone()))
        .collect()
}

/// In a one-step score of the daily and the weekly stream, each weekly row
/// was predicted after the daily row ending a day before it was assimilated,
/// and not the daily row ending with it; the score writes no `lead`. The
/// joint row at a stop carries the same value, and the first stop has
/// nothing assimilated before it.
#[test]
fn a_weekly_rows_assimilated_through_is_the_daily_row_before_it() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let folder = two_stream_one_step(&root, cwd.path());
    assert!(!header_of(&folder, "scored_rows.tsv").contains(&"lead".to_string()));
    let weekly = assimilated(&folder, "scored_rows.tsv", Some("weekly_cases"));
    assert_eq!(weekly.len(), 9, "{weekly:?}");
    for (stop, through) in &weekly {
        let stop: f64 = stop.parse().unwrap();
        assert_eq!(through.parse::<f64>().unwrap(), stop - 1.0, "the weekly row ending {stop}");
    }
    let daily = assimilated(&folder, "scored_rows.tsv", Some("daily_cases"));
    assert_eq!(daily[0], ("1".to_string(), String::new()), "nothing is assimilated before the first stop");
    assert_eq!(daily[1], ("2".to_string(), "1".to_string()));
    let joints = assimilated(&folder, "joint_rows.tsv", None);
    assert!(joints.contains(&("14".to_string(), "13".to_string())), "{joints:?}");
}

/// With the weekly stream alone, a row was predicted after the previous
/// week's row was assimilated. An `NA` row is not assimilated, so the row
/// after it was predicted from the week before; a held-out row is
/// assimilated after it is scored, so it counts for the rows after it. A
/// free-forward score writes the column empty.
#[test]
fn assimilated_through_skips_an_na_row_and_counts_a_held_out_one() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let weekly = std::fs::read_to_string(cwd.path().join("weekly_cases.tsv")).unwrap();
    let with_na: String = weekly.lines().map(|l| if l.starts_with("14\t") { "14\tNA\n".to_string() } else { format!("{l}\n") }).collect();
    assert_ne!(with_na, weekly, "the fixture's weekly row 14 moved");
    std::fs::write(cwd.path().join("weekly_na.tsv"), with_na).unwrap();
    let data = ["--data", "weekly_cases=weekly_na.tsv"];
    let base = ["predict", "model.camdl", "--params", "point.toml", "--n-draws", "2", "--holdout-after", "24"];
    let one_step = ["--mode", "one_step", "--particles", "40", "--label", "os"];
    ok(&predict_in(cwd.path(), &root, &[&base[..], &data, &one_step].concat()), "predict");
    ok(&score_in(cwd.path(), &root, &[&["@os"][..], &data].concat()), "score");
    let folder = the_score(&root);
    let rows = tsv_rows(&read(&folder, "scored_rows.tsv"));
    let through: Vec<(&str, &str, &str)> =
        rows.iter().map(|r| (r["t_stop"].as_str(), r["role"].as_str(), r["assimilated_through"].as_str())).collect();
    assert_eq!(
        through,
        [
            ("7", "trained", ""),
            ("21", "trained", "7"),
            ("28", "held_out", "21"),
            ("35", "held_out", "28"),
            ("42", "held_out", "35"),
            ("49", "held_out", "42"),
            ("56", "held_out", "49"),
            ("63", "held_out", "56"),
        ],
        "the NA row of day 14 is not scored and not assimilated; the held-out row of day 28 counts for day 35"
    );

    ok(&predict_in(cwd.path(), &root, &[&base[..], &data, &["--mode", "free_forward", "--label", "ff"]].concat()), "predict");
    ok(&score_in(cwd.path(), &root, &[&["@ff"][..], &data].concat()), "score");
    let ff = scores(&root).into_iter().map(|(p, _)| p).find(|p| *p != folder).unwrap();
    let rows = tsv_rows(&read(&ff, "scored_rows.tsv"));
    assert!(!rows.is_empty() && rows.iter().all(|r| r["mode"] == "free_forward" && r["assimilated_through"].is_empty()), "{rows:?}");
}
