//! gh#585 (Stage 3.1 of the 2026-08-29 honest-predictive-evaluation
//! proposal): a fit that declares `holdout_after` actually trains on the
//! truncated series, and writes the applied training window into
//! `fit.meta.json` — the §3.7.3(b) positive proof `camdl compare`'s
//! non-leakage gate reads. A rerun that hits the cache end-to-end must not
//! erase the recorded window (the sidecar is rewritten before the
//! short-circuit, without re-loading data).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

fn golden_ir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/seir_observations.ir.json")
}

fn find_fit_meta(root: &Path) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                return Some(p);
            }
        }
    }
    None
}

/// gh#585 (Stages 3.4/3.5): `camdl compare` scores holdout-declaring fits
/// out-of-sample — derives with `--score-from` at the sealed boundary,
/// verifies non-leakage, stamps `hold_out_tail` — refuses a mixed
/// comparison and a fit without the applied-window proof, and `--in-sample`
/// forces the old mode.
#[test]
fn compare_scores_holdout_fits_out_of_sample() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let write_fit = |name: &str, beta: f64, holdout_line: &str| {
        let sub = dir.join(name);
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("obs.tsv"),
            "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
        std::fs::write(sub.join("fit.toml"), format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data]
{holdout_line}
[data.observations]
weekly_cases = "obs.tsv"

[estimate.I0]
bounds = [1, 1000]
start  = 5

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = {beta}

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().canonicalize().unwrap().display())).unwrap();
        sub
    };

    let fit_a = write_fit("a", 0.10, "holdout_after = 21.0");
    let fit_b = write_fit("b", 0.15, "holdout_after = 21.0");
    let fit_c = write_fit("c", 0.10, "");

    for sub in [&fit_a, &fit_b, &fit_c] {
        let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["fit", "run", "fit.toml", "--seed", "1"])
            .current_dir(sub)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().expect("spawn camdl");
        assert!(out.status.success(), "fit run in {} failed:\nstderr={}",
            sub.display(), String::from_utf8_lossy(&out.stderr));
    }

    let compare = |paths: &[&str], extra: &[&str]| {
        let mut args = vec!["compare"];
        args.extend_from_slice(paths);
        args.extend_from_slice(&["--format", "json", "--particles", "100"]);
        args.extend_from_slice(extra);
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(&args)
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().expect("spawn camdl")
    };

    // Both fits sealed at t = 21 → held-out derivation, stamped and scored
    // only past the boundary.
    let out = compare(&["a/fit.toml", "b/fit.toml"], &[]);
    assert!(out.status.success(), "holdout compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("json output");
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["conditioning"]["hold_out_tail"]["train_end"],
            serde_json::json!(21.0),
            "row must be stamped hold_out_tail: {row}");
        assert_eq!(row["t_score"], serde_json::json!(2),
            "only t = 28, 35 lie past the sealed boundary: {row}");
    }

    // Mixed conditioning is refused — no override.
    let out = compare(&["a/fit.toml", "c/fit.toml"], &[]);
    assert!(!out.status.success(), "mixed conditioning must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("mix conditioning modes"),
        "must name the mixed modes:\n{stderr}");

    // --in-sample forces the old mode: full series, in_sample stamp.
    let out = compare(&["a/fit.toml", "b/fit.toml"], &["--in-sample"]);
    assert!(out.status.success(), "--in-sample compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["conditioning"], serde_json::json!("in_sample"));
        assert_eq!(row["t_score"], serde_json::json!(5));
    }

    // The §3.7.3(b) gate: strip the applied-window record from fit A's
    // sidecar (what a pre-gh#585 fit looks like — it declares a holdout it
    // never applied) and the hold_out_tail derivation must refuse.
    let meta_path = find_fit_meta(&fit_a.join("results")).unwrap();
    let mut meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    meta.as_object_mut().unwrap().remove("training_window");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    let out = compare(&["a/fit.toml", "b/fit.toml"], &[]);
    assert!(!out.status.success(),
        "a fit without the applied-window proof must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no applied training window"),
        "must name the missing proof and the fix:\n{stderr}");
}

/// gh#585 (Stage 3.2): `pfilter --score-from TIME` assimilates the full
/// series but scores the trace only at t > TIME, recording the boundary
/// (`score_from`, and `t0` as its index twin). The total log-likelihood is
/// unchanged — only the trace is windowed.
#[test]
fn pfilter_score_from_windows_the_trace_not_the_likelihood() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("obs.tsv"),
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
    std::fs::write(dir.join("theta.toml"),
        "sigma = 0.25\ngamma = 0.3\nrho = 0.5\nk = 10.0\np_detect = 0.5\n\
         N0 = 1000\nbeta = 0.1\nI0 = 5\n").unwrap();

    let run = |extra: &[&str], stem: &str| {
        let mut args = vec![
            "pfilter",
            golden_path(),
            "--data", "weekly_cases=obs.tsv",
            "--params", "theta.toml",
            "--particles", "200",
            "--dt", "1",
            "--seed", "1",
            "--save-prequential", stem,
        ];
        args.extend_from_slice(extra);
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(&args)
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    };

    let full = run(&[], "full");
    assert!(full.status.success(), "plain pfilter failed:\nstderr={}",
        String::from_utf8_lossy(&full.stderr));
    let windowed = run(&["--score-from", "21"], "tail");
    assert!(windowed.status.success(), "--score-from pfilter failed:\nstderr={}",
        String::from_utf8_lossy(&windowed.stderr));

    // Same filter pass: identical total log-likelihood on stdout.
    assert_eq!(
        String::from_utf8_lossy(&full.stdout).trim(),
        String::from_utf8_lossy(&windowed.stdout).trim(),
        "--score-from must not change the total log-likelihood");

    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(
            &std::fs::read_to_string(dir.join(name)).unwrap()).unwrap()
    };
    let full_trace = read("full.json");
    let tail_trace = read("tail.json");
    assert_eq!(full_trace["steps"].as_array().unwrap().len(), 5);
    assert_eq!(full_trace["score_from"], serde_json::Value::Null);

    // t = 7, 14, 21 assimilated but not scored; t = 28, 35 scored.
    assert_eq!(tail_trace["t0"], serde_json::json!(3));
    assert_eq!(tail_trace["score_from"], serde_json::json!(21.0));
    let steps = tail_trace["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2, "only the two post-boundary steps are scored");
    assert_eq!(steps[0]["t"], serde_json::json!(28.0));
    // Assimilation makes the windowed scores CONDITIONAL on the earlier
    // observations: the scored step must be identical to the same step of
    // the full trace (same filter, same seed, same draws).
    assert_eq!(steps[0]["log_score"],
        full_trace["steps"][3]["log_score"],
        "the scored tail must be the same filter pass, windowed");

    // A boundary at/after the last observation scores nothing — refused.
    let bad = run(&["--score-from", "35"], "none");
    assert!(!bad.status.success(), "--score-from at last obs must be refused");
    assert!(String::from_utf8_lossy(&bad.stderr).contains("nothing"),
        "must say nothing would be scored");
}

fn golden_path() -> &'static str {
    // Leaked once per test binary: Command::args wants &str lifetimes.
    Box::leak(golden_ir().canonicalize().unwrap()
        .to_string_lossy().into_owned().into_boxed_str())
}

#[test]
fn fit_run_applies_holdout_and_records_the_training_window() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    std::fs::write(dir.join("obs.tsv"),
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
    let toml_src = format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data]
holdout_after = 21.0

[data.observations]
weekly_cases = "obs.tsv"

[estimate.I0]
bounds = [1, 1000]
start  = 5

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = 0.1

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().canonicalize().unwrap().display());
    std::fs::write(dir.join("fit.toml"), toml_src).unwrap();

    let run = || {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["fit", "run", "fit.toml", "--seed", "1"])
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    };

    let out = run();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "fit run with holdout_after failed:\nstdout={}\nstderr={stderr}",
        String::from_utf8_lossy(&out.stdout));

    // The banner names the training window and what was withheld.
    assert!(stderr.contains("holdout:") && stderr.contains("withheld"),
        "the fit banner must print the training window:\n{stderr}");

    // The §3.7.3(b) proof: fit.meta.json carries the applied window.
    let meta_path = find_fit_meta(&dir.join("results"))
        .expect("fit run wrote a fit.meta.json under results/");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["training_window"]["train_end"], serde_json::json!(21.0),
        "fit.meta.json must record the applied training window: {meta}");

    // An all-cache-hit rerun rewrites the sidecar without re-loading data —
    // the recorded proof must survive (sticky, like `label`).
    let out2 = run();
    assert!(out2.status.success(), "cache-hit rerun failed:\nstderr={}",
        String::from_utf8_lossy(&out2.stderr));
    let meta2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta2["training_window"]["train_end"], serde_json::json!(21.0),
        "a cache-hit rerun must not erase the recorded training window");
}

/// A two-stream fit (`weekly_cases` every 7 days, `detection` every 14) with
/// a `[data.holdout]` file for `weekly_cases` only: training t = 7, 14, 21
/// and held-out rows t = 28, 35. `detection`'s training file is the caller's,
/// so a test can make it end before or after the held-out rows.
fn write_partial_holdout_fit(sub: &Path, beta: f64, detection_tsv: &str) {
    std::fs::create_dir_all(sub).unwrap();
    std::fs::write(sub.join("cases.tsv"),
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n").unwrap();
    std::fs::write(sub.join("cases_holdout.tsv"),
        "time\tweekly_cases\n28\t4\n35\t5\n").unwrap();
    std::fs::write(sub.join("detection.tsv"), detection_tsv).unwrap();
    std::fs::write(sub.join("fit.toml"), format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "cases.tsv"
detection = "detection.tsv"

[data.holdout]
weekly_cases = "cases_holdout.tsv"

[estimate.I0]
bounds = [1, 1000]
start  = 5

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = {beta}

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, ir = golden_ir().canonicalize().unwrap().display())).unwrap();
}

fn fit_run(bin: &Path, sub: &Path) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(["fit", "run", "fit.toml", "--seed", "1"])
        .current_dir(sub)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// `[data.holdout]` records `train_end` as the latest training time over
/// every bound stream, not only the streams that have a holdout file. A
/// stream without one whose training runs past another stream's held-out
/// rows would be scored by `compare` as held out although θ was fitted to
/// it, so the fit refuses it and names that stream.
#[test]
fn partial_holdout_refuses_training_data_inside_the_held_out_window() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();

    // `detection` trains at t = 14, 28, 42: past the first held-out row of
    // `weekly_cases` (t = 28).
    let leaky = tmp.path().join("leaky");
    write_partial_holdout_fit(&leaky, 0.10, "time\tdetection\n14\t1\n28\t0\n42\t1\n");
    let out = fit_run(&bin, &leaky);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "a stream training past the held-out rows must be refused:\n{stderr}");
    assert!(stderr.contains("'detection'") && stderr.contains("holdout_after"),
        "the error must name the stream whose training runs past the \
         held-out rows, and the fix:\n{stderr}");

    // Streams may end training at different times as long as every held-out
    // row lies after the latest: `detection` ends at t = 14, `weekly_cases`
    // at t = 21, so the recorded window ends at t = 21.
    let ok = tmp.path().join("ok");
    write_partial_holdout_fit(&ok, 0.10, "time\tdetection\n14\t1\n");
    let out = fit_run(&bin, &ok);
    assert!(out.status.success(), "a valid partial holdout must fit:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let meta_path = find_fit_meta(&ok.join("results")).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["training_window"]["train_end"], serde_json::json!(21.0),
        "train_end is the latest training time over all streams: {meta}");
}

/// A stored `[data.holdout]` fit whose training data runs past its recorded
/// `train_end` — what a fit run before `train_end` covered every bound
/// stream looks like — is refused the `hold_out_tail` stamp by `compare`
/// rather than scored: the rows past `train_end` in a training file are
/// data θ was fitted to.
#[test]
fn compare_refuses_holdout_stamp_when_training_runs_past_train_end() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Two valid partial-holdout fits: `detection` ends training at t = 14,
    // before the recorded train_end = 21.
    let fits = [dir.join("a"), dir.join("b")];
    for (sub, beta) in fits.iter().zip([0.10, 0.15]) {
        write_partial_holdout_fit(sub, beta, "time\tdetection\n14\t1\n");
        let out = fit_run(&bin, sub);
        assert!(out.status.success(), "fit run in {} failed:\nstderr={}",
            sub.display(), String::from_utf8_lossy(&out.stderr));
    }
    let compare = || {
        Command::new(&bin).env_remove("CLICOLOR_FORCE")
            .args(["compare", "a/fit.toml", "b/fit.toml",
                   "--format", "json", "--particles", "100", "--replicates", "1"])
            .current_dir(dir)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().expect("spawn camdl")
    };

    // A valid partial holdout is scored held-out: only weekly_cases' t = 28
    // and 35 lie past train_end = 21.
    let out = compare();
    assert!(out.status.success(), "valid partial-holdout compare failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json output");
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["conditioning"]["hold_out_tail"]["train_end"],
            serde_json::json!(21.0), "row must be stamped hold_out_tail: {row}");
        assert_eq!(row["t_score"], serde_json::json!(2), "{row}");
    }

    // The defective shape: `detection`'s training file now runs to t = 42,
    // past the recorded train_end = 21. Its rows at t = 28 and 42 would be
    // scored as held out although they are training data.
    for sub in &fits {
        let mut f = std::fs::OpenOptions::new()
            .append(true).open(sub.join("detection.tsv")).unwrap();
        std::io::Write::write_all(&mut f, b"28\t0\n42\t1\n").unwrap();
    }
    let out = compare();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(),
        "training data past train_end must refuse the hold_out_tail stamp:\n{stderr}");
    assert!(stderr.contains("'detection'") && stderr.contains("Re-run")
            && stderr.contains("--in-sample"),
        "the refusal must name the stream and both fixes:\n{stderr}");
}
