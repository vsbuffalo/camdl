//! `camdl compare` on stored scores, checked by running the release binary
//! on this module's fixture: a held-out comparison of two stored fits is the
//! difference of their stored scores, a score compare computes is stored and
//! reused, and scores at different leads or reaches are refused.

use super::*;

fn ok(out: &Output, what: &str) -> String {
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{what}:\n{err}");
    err
}

fn run(cwd: &Path, root: &Path, line: &[&str]) -> Output {
    camdl(cwd, root, &line.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[])
}

/// The fixture's three outcome files, as `--data` arguments.
const DATA: &[&str] =
    &["--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv", "--data", "survey=survey.tsv"];

/// Every stored score under `root`: its folder.
fn score_folders(root: &Path) -> Vec<PathBuf> {
    folders(root).into_iter().filter(|(_, id, _)| id["kind"] == crate::fit_store::Score::KIND).map(|(p, _, _)| p).collect()
}

/// The sum of a score's `joint` log scores over its rows of `mode` and
/// `role`: the elpd compare reports for it.
fn joint_sum(folder: &Path, mode: &str, role: &str) -> (usize, f64) {
    let rows = tsv_rows(&std::fs::read_to_string(folder.join("scored.tsv")).unwrap());
    let joint: Vec<f64> = rows
        .iter()
        .filter(|r| r["stream"] == "joint" && r["mode"] == mode && r["role"] == role)
        .map(|r| r["log_score"].parse::<f64>().unwrap())
        .collect();
    (joint.len(), joint.iter().sum())
}

/// The rows of `compare --format json`, by name.
fn compared(out: &Output) -> BTreeMap<String, serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
    });
    doc["rows"].as_array().unwrap().iter().map(|r| (r["name"].as_str().unwrap().to_string(), r.clone())).collect()
}

/// Two stored fits that held their rows after day 24 out, each predicted
/// one-step and scored by `camdl score` against the files they were given:
/// `compare` finds the two stored scores, computes nothing, and reports each
/// row's elpd as the sum of its score's joint log scores over the held-out
/// rows, and Δelpd as their difference.
#[test]
fn a_held_out_compare_of_two_stored_fits_differences_their_stored_scores() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let opts = ["--mode", "one_step", "--n-draws", "4", "--particles", "30"];
    for (fit, label) in [("@fit", "p_fit"), ("@other", "p_other")] {
        ok(&predict_in(cwd.path(), &root, &[&["predict", fit][..], &opts, &["--label", label]].concat()), "predict");
        ok(&run(cwd.path(), &root, &[&["score", &format!("@{label}")][..], DATA].concat()), "score");
    }
    let before = folders(&root).len();
    let out = run(cwd.path(), &root, &[
        "compare", "@fit", "@other", "--mode", "one_step", "--draws", "4", "--particles", "30", "--replicates", "1",
        "--format", "json",
    ]);
    ok(&out, "compare");
    assert_eq!(folders(&root).len(), before, "compare found the stored scores and stored nothing");
    let rows = compared(&out);
    let mut stored: Vec<(usize, f64)> =
        score_folders(&root).iter().map(|s| joint_sum(s, "one_step", "held_out")).collect();
    let mut shown: Vec<(usize, f64)> = rows
        .values()
        .map(|r| (r["t_score"].as_u64().unwrap() as usize, r["elpd"].as_f64().unwrap()))
        .collect();
    stored.sort_by(|a, b| a.1.total_cmp(&b.1));
    shown.sort_by(|a, b| a.1.total_cmp(&b.1));
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1.0);
    assert!(
        shown.len() == 2 && shown.iter().zip(&stored).all(|(s, t)| s.0 == t.0 && close(s.1, t.1)),
        "each row is one stored score's held-out rows: {shown:?} vs {stored:?}"
    );
    assert!(stored.iter().all(|(n, _)| *n > 0), "{stored:?}");
    for r in rows.values() {
        assert_eq!(r["role"], "held_out", "{r}");
        assert!(r["conditioning"]["hold_out_tail"]["train_end"] == serde_json::json!(24.0), "{r}");
    }
    let base = doc_baseline(&out);
    let other = rows.keys().find(|k| **k != base).unwrap().clone();
    let delta = rows[&other]["delta_elpd"].as_f64().unwrap();
    let want = rows[&other]["elpd"].as_f64().unwrap() - rows[&base]["elpd"].as_f64().unwrap();
    assert!(close(delta, want), "Δelpd {delta} is the difference of the stored sums, {want}");
    assert!(close(want, stored[0].1 - stored[1].1), "the baseline is the higher elpd");
}

fn doc_baseline(out: &Output) -> String {
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    doc["baseline"].as_str().unwrap().to_string()
}

/// A score compare needs and does not find is computed, from a prediction
/// it computes, and stored; a second compare of the same fits finds both
/// and computes nothing, and prints the same table.
#[test]
fn a_missing_score_is_computed_once_and_reused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let line = [&["compare", "@fit", "@other", "--draws", "3", "--particles", "20", "--replicates", "1"][..], DATA].concat();
    let start = folders(&root).len();
    let first = run(cwd.path(), &root, &line);
    ok(&first, "first compare");
    assert_eq!(score_folders(&root).len(), 2, "one score per fit");
    assert_eq!(predictions(&root).len(), 2, "one prediction per fit");
    let after_first = folders(&root).len();
    assert!(after_first > start);
    let second = run(cwd.path(), &root, &line);
    let err = ok(&second, "second compare");
    assert_eq!(folders(&root).len(), after_first, "nothing is stored again");
    assert!(err.contains("reused the stored score"), "{err}");
    assert_eq!(first.stdout, second.stdout, "the same table");
    let table = String::from_utf8_lossy(&first.stdout);
    assert!(table.contains("role") && table.contains("held_out"), "{table}");
    assert!(table.contains("mean WIS") && table.contains("cov50"), "{table}");
    assert!(table.contains("filter noise not measured") || table.contains("within noise"), "{table}");
}

/// The held-out rows of a stored fit are scored only against outcome files
/// or a stored score: with neither, compare says so and stores no score.
/// `--in-sample` compares the training rows, which need no file.
#[test]
fn held_out_rows_need_their_outcomes_and_in_sample_rows_do_not() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let base = ["compare", "@fit", "@other", "--draws", "3", "--particles", "20", "--replicates", "1"];
    let out = run(cwd.path(), &root, &base);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("--data STREAM=FILE") && err.contains("--in-sample"), "{err}");
    assert!(score_folders(&root).is_empty());
    let out = run(cwd.path(), &root, &[&base[..], &["--in-sample", "--format", "json"]].concat());
    ok(&out, "compare --in-sample");
    let rows = compared(&out);
    for r in rows.values() {
        assert_eq!(r["role"], "trained", "{r}");
        assert_eq!(r["conditioning"], "in_sample", "{r}");
        assert!(r["derivation"]["overrides"].as_array().unwrap().contains(&serde_json::json!("in_sample_forced")), "{r}");
    }
}

/// Forecast scores are compared lead by lead, one table each, and two
/// forecasts that reach to different leads, or whose reach is written
/// differently, are refused, naming the field.
#[test]
fn forecasts_are_compared_per_lead_and_refused_at_different_leads_or_reaches() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let fc = |fit: &str, horizon: &str, label: &str| {
        ok(
            &predict_in(cwd.path(), &root, &[
                "predict", fit, "--horizon", horizon, "--stream", "daily_cases", "--n-draws", "3", "--particles", "20",
                "--label", label,
            ]),
            "predict",
        );
        ok(&run(cwd.path(), &root, &[&["score", &format!("@{label}")][..], DATA].concat()), "score");
    };
    fc("@fit", "3 days", "a3");
    fc("@other", "3 days", "b3");
    fc("@other", "5 days", "b5");
    let out = run(cwd.path(), &root, &["compare", "@a3", "@b3"]);
    let table = ok(&out, "compare per lead");
    let _ = table;
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(text.matches("Scored steps:").count(), 3, "one table per lead:\n{text}");
    assert!(text.contains("lead") && text.contains("future"), "{text}");

    let out = run(cwd.path(), &root, &["compare", "@a3", "@b5"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("different leads"), "{err}");

    // The same reach in model time, written as a bare number.
    fc("@other", "3", "b3n");
    let out = run(cwd.path(), &root, &["compare", "@a3", "@b3n"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("(reach)") && err.contains("\"3 days\"") && err.contains("\"3\""), "{err}");
}

/// Each fixture data file cut to its header and the rows after day 24, the
/// training extent of `@fit`: the files `--data-held-out` takes.
fn write_held_out_files(cwd: &Path) {
    for f in ["daily_cases.tsv", "weekly_cases.tsv", "survey.tsv"] {
        let text = std::fs::read_to_string(cwd.join(f)).unwrap();
        let mut lines = text.lines();
        let mut out = format!("{}\n", lines.next().unwrap());
        for l in lines.filter(|l| l.split('\t').next().unwrap().parse::<f64>().unwrap() > 24.0) {
            out += &format!("{l}\n");
        }
        std::fs::write(cwd.join(format!("held_{f}")), out).unwrap();
    }
}

const HELD: &[&str] = &[
    "--data-held-out", "daily_cases=held_daily_cases.tsv", "--data-held-out", "weekly_cases=held_weekly_cases.tsv",
    "--data-held-out", "survey=held_survey.tsv",
];

/// A stream's held-out rows given alone (`--data-held-out`) are the outcome
/// the whole file gives: the score of a prediction against the whole files is
/// found again, not computed, when the same rows are given as held-out files,
/// and `compare` reads them too. A file with a row that does not end after
/// the conditioning rows is refused, naming the row, and so is a stream given
/// both ways.
#[test]
fn held_out_rows_given_alone_are_the_outcome_the_whole_file_gives() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    write_held_out_files(cwd.path());
    ok(&predict_in(cwd.path(), &root, &[
        "predict", "@fit", "--mode", "one_step", "--n-draws", "3", "--particles", "20", "--label", "p",
    ]), "predict");
    ok(&run(cwd.path(), &root, &[&["score", "@p"][..], DATA].concat()), "score of the whole files");
    assert_eq!(score_folders(&root).len(), 1);
    let err = ok(&run(cwd.path(), &root, &[&["score", "@p"][..], HELD].concat()), "score of the held-out files");
    assert!(err.contains("reused the stored score"), "one outcome stated two ways is one score: {err}");
    assert_eq!(score_folders(&root).len(), 1);

    let out = run(cwd.path(), &root, &[
        "score", "@p", "--data", "daily_cases=daily_cases.tsv", "--data-held-out", "daily_cases=held_daily_cases.tsv",
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("--data and --data-held-out both give the outcomes of stream `daily_cases`"), "{err}");

    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let early: String = daily.lines().enumerate().filter(|(i, l)| *i == 0 || l.split('\t').next().unwrap().parse::<f64>().unwrap() >= 24.0)
        .map(|(_, l)| format!("{l}\n")).collect();
    std::fs::write(cwd.path().join("early.tsv"), early).unwrap();
    let out = run(cwd.path(), &root, &["score", "@p", "--data-held-out", "daily_cases=early.tsv"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("row 1 of 'daily_cases' (time 24) ends at t = 24"), "{err}");
    assert_eq!(score_folders(&root).len(), 1, "nothing more is stored");

    let line = [&["compare", "@fit", "@other", "--draws", "3", "--particles", "20", "--replicates", "1"][..], HELD].concat();
    ok(&run(cwd.path(), &root, &line), "compare --data-held-out");

    // What the store records of the outcome tables names the files given,
    // not the scratch files a held-out file is completed into.
    let text = store_text(&root);
    assert!(!text.contains("camdl-score-held-out-"), "a scratch path is recorded");
    assert!(text.contains("held_daily_cases.tsv"), "the held-out file given is recorded");
}

/// Every text file under `root`, concatenated.
fn store_text(root: &Path) -> String {
    let mut text = String::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(t) = std::fs::read_to_string(&path) {
                text += &t;
            }
        }
    }
    text
}

/// Every prediction compare makes runs its filters at one particle count,
/// `--particles` or 1000. A stored prediction at another count is refused
/// before anything is computed, naming each input's count and the
/// `--particles` that agrees with it; given that count, the stored score is
/// compared against a fit run, each predicted at five seeds by default, the
/// stored one posed again from its own seed.
#[test]
fn one_particle_count_for_every_prediction_and_a_stored_one_at_another_is_refused_first() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(&predict_in(cwd.path(), &root, &[
        "predict", "@fit", "--mode", "one_step", "--n-draws", "3", "--particles", "20", "--label", "p20",
    ]), "predict");
    ok(&run(cwd.path(), &root, &[&["score", "@p20"][..], DATA].concat()), "score");
    let before = folders(&root).len();
    let out = run(cwd.path(), &root, &[&["compare", "@p20", "@other", "--draws", "3"][..], DATA].concat());
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("'@p20' 20 (stored)") && err.contains("'@other' 1000 (a fit run"), "{err}");
    assert!(err.contains("pass --particles 20"), "{err}");
    assert_eq!(folders(&root).len(), before, "refused before anything is computed");

    let out = run(cwd.path(), &root, &[&["compare", "@p20", "@other", "--draws", "3", "--particles", "20", "--format", "json"][..], DATA].concat());
    ok(&out, "compare at the stored count");
    for r in compared(&out).values() {
        assert_eq!(r["derivation"]["particles"], 20, "{r}");
        assert_eq!(r["derivation"]["replicates"], 5, "{r}");
        assert!(r["mc_se_elpd"].as_f64().is_some_and(f64::is_finite), "{r}");
    }
    assert_eq!(predictions(&root).len(), 10, "five predictions of each input");
}

/// A fit run is compared against a stored score with no waiver: the stored
/// score's prediction ran its filters at compare's count, and both are
/// predicted at five seeds.
#[test]
fn a_fit_run_compares_against_a_stored_score_with_no_waiver() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    ok(&predict_in(cwd.path(), &root, &[
        "predict", "@fit", "--mode", "one_step", "--n-draws", "3", "--particles", "1000", "--label", "p",
    ]), "predict");
    ok(&run(cwd.path(), &root, &[&["score", "@p", "--label", "s"][..], DATA].concat()), "score");
    let out = run(cwd.path(), &root, &[&["compare", "@s", "@other", "--draws", "3", "--format", "json"][..], DATA].concat());
    ok(&out, "compare");
    let rows = compared(&out);
    assert_eq!(rows.len(), 2);
    for r in rows.values() {
        assert_eq!(r["derivation"]["replicates"], 5, "{r}");
        assert_eq!(r["role"], "held_out", "{r}");
    }
}

/// An outcome file that no compared prediction reads is refused, naming
/// its stream, and so is a set of files that gives no row past the rows the
/// predictions conditioned on.
#[test]
fn an_outcome_file_that_answers_nothing_is_refused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let base = ["compare", "@fit", "@other", "--draws", "3", "--particles", "20", "--replicates", "1"];
    let out = run(cwd.path(), &root, &[&base[..], &["--data", "kases=daily_cases.tsv"]].concat());
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("--data kases=…: no compared prediction binds a stream or stream source `kases`"), "{err}");
    assert!(err.contains("`daily_cases`"), "{err}");
    assert!(score_folders(&root).is_empty() && predictions(&root).is_empty(), "nothing is computed");

    let daily = std::fs::read_to_string(cwd.path().join("daily_cases.tsv")).unwrap();
    let training: String = daily.lines().enumerate()
        .filter(|(i, l)| *i == 0 || l.split('\t').next().unwrap().parse::<f64>().unwrap() <= 24.0)
        .map(|(_, l)| format!("{l}\n")).collect();
    std::fs::write(cwd.path().join("training.tsv"), training).unwrap();
    let out = run(cwd.path(), &root, &[&base[..], &["--data", "daily_cases=training.tsv"]].concat());
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("the outcome files give no one_step row past the rows its prediction conditioned on")
        && err.contains("--in-sample"), "{err}");
}
