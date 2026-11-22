//! `camdl compare` on fits whose `[data]` keys are not the stream names the
//! scores carry: an indexed block (`cases[p in patch]` → leaves `cases_urban`,
//! `cases_rural`, both read from the source `cases`) and blocks that share one
//! source through `from <label>`.
//!
//! A derived row is scored by a `camdl pfilter` child, so every test runs two
//! small IF2 fits, stores them in the test's store labelled `a` and `b`, and
//! compares them by label.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn binary() -> PathBuf {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} - run `make build-rust`",
        bin.display());
    bin
}

fn camdlc() -> PathBuf {
    let bin = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(bin.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", bin.display());
    bin
}

fn two_patch_ir() -> PathBuf {
    manifest().join("../../../ocaml/golden/sir_two_patch_long_obs.ir.json")
        .canonicalize().unwrap()
}

/// Run `camdl` in `dir`, where `dir` is a test's root or one of its fit
/// directories; every fit of a test is stored in `<root>/store`.
fn camdl(dir: &Path, args: &[&str]) -> Output {
    let root = if dir.join("fit.toml").is_file() { dir.parent().unwrap() } else { dir };
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_STORE", root.join("store"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .args(args)
        .current_dir(dir)
        .output().expect("spawn camdl")
}

const METHOD: &str = r#"
[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 50
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#;

/// Long-form `(time, patch, cases)` training rows for both patches.
const TWO_PATCH_TRAIN: &str = "time\tpatch\tcases\n\
    7\turban\t12\n7\trural\t3\n14\turban\t18\n14\trural\t6\n\
    21\turban\t25\n21\trural\t9\n28\turban\t20\n28\trural\t7\n";
const TWO_PATCH_HOLDOUT: &str = "time\tpatch\tcases\n\
    35\turban\t15\n35\trural\t5\n42\turban\t11\n42\trural\t4\n";

/// Write `dir/name/fit.toml` from `data` (its `[data]` tables) and the files
/// in `files`, and run the fit there, stored in `dir/store` under the label
/// `name`.
fn fit(dir: &Path, name: &str, model: &Path, data: &str, rest: &str,
       files: &[(&str, &str)]) -> PathBuf {
    let sub = dir.join(name);
    std::fs::create_dir_all(&sub).unwrap();
    for (f, text) in files {
        std::fs::write(sub.join(f), text).unwrap();
    }
    std::fs::write(sub.join("fit.toml"), format!(
        "store = \"../store\"\n\n[model]\ncamdl = \"{}\"\n\n{data}\n{rest}\n{METHOD}",
        model.display())).unwrap();
    let out = camdl(&sub, &["fit", "run", "fit.toml", "--seed", "1", "--label", name]);
    assert!(out.status.success(), "fit run in {} failed:\n{}",
        sub.display(), String::from_utf8_lossy(&out.stderr));
    sub
}

const TWO_PATCH_PARAMS: &str = r#"
[estimate.gamma]
bounds = [0.01, 1.0]
start  = 0.1

[fixed]
rho = 0.6
k   = 5.0
"#;

fn two_patch_fit(dir: &Path, name: &str, data: &str, beta: f64,
                 files: &[(&str, &str)]) -> PathBuf {
    fit(dir, name, &two_patch_ir(), data,
        &format!("{TWO_PATCH_PARAMS}beta = {beta}\n"), files)
}

fn compare(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["compare", "@a", "@b",
        "--particles", "100", "--draws", "1", "--replicates", "1"];
    args.extend_from_slice(extra);
    camdl(dir, &args)
}

fn json(out: &Output) -> serde_json::Value {
    assert!(out.status.success(), "compare failed:\n{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("compare --format json")
}

/// Every derived row scores both leaves of the indexed block, and the data
/// checks cover the one source they read.
fn assert_scored_and_checked(v: &serde_json::Value) {
    assert!(v["data_unchecked"].is_null(), "every row's data was checked:\n{v}");
    assert!(v["data_change_unchecked"].is_null(), "every file was checked:\n{v}");
    for row in v["rows"].as_array().unwrap() {
        let streams = row["derivation"]["replicate_elpd_by_stream"].as_object()
            .map(|m| m.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        assert!(streams.iter().any(|s| s == "cases_urban")
                && streams.iter().any(|s| s == "cases_rural"),
            "row {} scores both leaves: {row}", row["name"]);
    }
}

/// `[data] file = ...` on a model whose one block expands to two leaves.
#[test]
fn a_shorthand_fit_whose_block_expands_to_leaves_is_scored_and_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n";
    two_patch_fit(dir, "a", data, 0.05, &[("train.tsv", TWO_PATCH_TRAIN)]);
    two_patch_fit(dir, "b", data, 0.06, &[("train.tsv", TWO_PATCH_TRAIN)]);
    assert_scored_and_checked(&json(&compare(dir, &["--format", "json"])));
}

/// The shorthand with an explicit `[data.holdout]`, which is keyed by source:
/// the holdout reaches both leaves of the block. A stored fit keeps its
/// held-out rows' periods and not their values, so `compare` refuses to
/// score them and counts each leaf's held-out rows; with `--in-sample` it
/// scores both leaves' training rows and checks their data.
#[test]
fn a_shorthand_holdout_fit_whose_block_expands_to_leaves_holds_out_both_leaves() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    let files = [("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)];
    two_patch_fit(dir, "a", data, 0.05, &files);
    two_patch_fit(dir, "b", data, 0.06, &files);
    let out = compare(dir, &["--format", "json"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "a held-out score is refused:\n{stderr}");
    assert!(stderr.contains("held rows out of its training data (cases_rural: 2, cases_urban: 2)")
            && stderr.contains("--in-sample"), "{stderr}");
    let v = json(&compare(dir, &["--format", "json", "--in-sample"]));
    assert_scored_and_checked(&v);
    for row in v["rows"].as_array().unwrap() {
        assert_eq!(row["conditioning"], "in_sample", "scored on the training rows: {row}");
    }
}

/// Two blocks that read one file through `from surv`: `[data.observations]`
/// binds the source `surv`, which is no leaf's name and no prefix of one.
const SHARED_SOURCE_MODEL: &str = r#"
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

simulate {
  from = 0 'days
  to   = 60 'days
}
"#;

const SURV: &str = "time\tcases\trecs\n7\t3\t1\n14\t6\t3\n21\t9\t5\n28\t8\t7\n35\t6\t6\n";

#[test]
fn a_fit_binding_a_shared_from_source_is_scored_and_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let model = dir.join("shared.camdl");
    std::fs::write(&model, SHARED_SOURCE_MODEL).unwrap();
    let rest = |beta: f64| format!(
        "[estimate.gamma]\nbounds = [0.01, 1.0]\nstart = 0.2\n\n\
         [fixed]\nbeta = {beta}\nrho = 0.5\nN0 = 1000\nI0 = 5\nk = 2.0\n");
    let data = "[data.observations]\nsurv = \"surv.tsv\"\n";
    fit(dir, "a", &model, data, &rest(0.4), &[("surv.tsv", SURV)]);
    fit(dir, "b", &model, data, &rest(0.5), &[("surv.tsv", SURV)]);
    let v = json(&compare(dir, &["--format", "json"]));
    assert!(v["data_unchecked"].is_null() && v["data_change_unchecked"].is_null(),
        "every row's data was checked:\n{v}");
}

/// Two `[data.holdout]` fits on identical training files whose holdout files
/// differ. `compare --in-sample` scores only the training rows, which are the
/// same in both, so the comparison is made and its data checks pass.
#[test]
fn holdout_fits_whose_holdout_files_differ_compare_on_their_training_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    two_patch_fit(dir, "a", data, 0.05,
        &[("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)]);
    let revised = TWO_PATCH_HOLDOUT.replace("42\turban\t11", "42\turban\t30");
    two_patch_fit(dir, "b", data, 0.05,
        &[("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", &revised)]);
    let v = json(&compare(dir, &["--format", "json", "--in-sample"]));
    assert_scored_and_checked(&v);
    assert!(v["data_changed"].is_null(), "the training rows scored are the same:\n{v}");
}

/// `compare` reads a `[data.holdout]` fit's data from its stored problem
/// (`inputs/` and `held_out/<stream>.tsv`), not from the files the fit file
/// names: editing the holdout file after the fit ran changes nothing
/// `compare --in-sample` reports, and its data checks flag no change.
#[test]
fn an_edited_holdout_file_does_not_change_what_compare_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    let files = [("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)];
    let a = two_patch_fit(dir, "a", data, 0.05, &files);
    two_patch_fit(dir, "b", data, 0.06, &files);
    let scores = |v: &serde_json::Value| -> Vec<(String, f64)> {
        assert!(v["data_changed"].is_null() && v["data_change_unchecked"].is_null(),
            "the holdout file was checked and is unchanged:\n{v}");
        v["rows"].as_array().unwrap().iter()
            .map(|r| (r["name"].as_str().unwrap().to_string(), r["elpd"].as_f64().unwrap()))
            .collect()
    };
    let before = scores(&json(&compare(dir, &["--format", "json", "--in-sample"])));

    let held_out = fit_store_support::problems(&dir.join("store")).into_iter()
        .filter_map(|p| std::fs::read_dir(p.join("held_out")).ok())
        .flat_map(|d| d.flatten().map(|e| e.path()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert!(!held_out.is_empty(), "each problem stores its held-out rows");

    std::fs::write(a.join("holdout.tsv"), TWO_PATCH_HOLDOUT.replace("42\turban\t11", "42\turban\t30"))
        .unwrap();
    let after = scores(&json(&compare(dir, &["--format", "json", "--in-sample"])));
    assert_eq!(before, after, "the edit must not change what compare scores");
}
