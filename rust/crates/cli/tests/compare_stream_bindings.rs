//! `camdl compare` on fits whose `[data]` keys are not the stream names the
//! scores carry: an indexed block (`cases[p in patch]` → leaves `cases_urban`,
//! `cases_rural`, both read from the source `cases`) and blocks that share one
//! source through `from <label>`.
//!
//! A derived row is scored by a `camdl pfilter` child, so every test runs two
//! small IF2 fits and compares them by `fit.toml` path.

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

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
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
/// in `files`, and run the fit there.
fn fit(dir: &Path, name: &str, model: &Path, data: &str, rest: &str,
       files: &[(&str, &str)]) -> PathBuf {
    let sub = dir.join(name);
    std::fs::create_dir_all(&sub).unwrap();
    for (f, text) in files {
        std::fs::write(sub.join(f), text).unwrap();
    }
    std::fs::write(sub.join("fit.toml"), format!(
        "output_dir = \"results\"\n\n[model]\ncamdl = \"{}\"\n\n{data}\n{rest}\n{METHOD}",
        model.display())).unwrap();
    let out = camdl(&sub, &["fit", "run", "fit.toml", "--seed", "1"]);
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
    let mut args = vec!["compare", "a/fit.toml", "b/fit.toml",
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

/// The shorthand with an explicit `[data.holdout]`, which is keyed by source.
#[test]
fn a_shorthand_holdout_fit_whose_block_expands_to_leaves_is_scored_held_out() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    let files = [("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)];
    two_patch_fit(dir, "a", data, 0.05, &files);
    two_patch_fit(dir, "b", data, 0.06, &files);
    let v = json(&compare(dir, &["--format", "json"]));
    for row in v["rows"].as_array().unwrap() {
        assert!(row["conditioning"].get("hold_out_tail").is_some(),
            "scored held out: {row}");
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

fn fit_meta(fit_dir: &Path) -> PathBuf {
    let mut stack = vec![fit_dir.join("results")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                return p;
            }
        }
    }
    panic!("no fit.meta.json under {}", fit_dir.display());
}

/// Rewrite one field of a fit's sidecar; `Null` removes it, as a sidecar
/// written before the field existed has none.
fn rewrite_sidecar(fit_dir: &Path, field: &str, value: serde_json::Value) {
    let path = fit_meta(fit_dir);
    let mut meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    if value.is_null() {
        meta.as_object_mut().unwrap().remove(field).expect("the field was recorded");
    } else {
        meta[field] = value;
    }
    std::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// A sidecar written before `data_hashes` was keyed by source records a
/// shorthand fit's file once per leaf. Where every leaf records the same
/// hash it names one file and is read as that source's; where the leaves
/// disagree no one hash names the file, and the source is reported as not
/// checked, with the reason, rather than compared or passed.
#[test]
fn an_old_leaf_keyed_sidecar_is_read_by_source_or_reported_unchecked() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n";
    let a = two_patch_fit(dir, "a", data, 0.05, &[("train.tsv", TWO_PATCH_TRAIN)]);
    two_patch_fit(dir, "b", data, 0.06, &[("train.tsv", TWO_PATCH_TRAIN)]);
    let h = sha256_hex(TWO_PATCH_TRAIN.as_bytes())[..8].to_string();

    rewrite_sidecar(&a, "data_hashes", serde_json::json!({"cases_urban": h, "cases_rural": h}));
    assert_scored_and_checked(&json(&compare(dir, &["--format", "json"])));

    rewrite_sidecar(&a, "data_hashes",
        serde_json::json!({"cases_urban": h, "cases_rural": "0badf00d"}));
    let v = json(&compare(dir, &["--format", "json"]));
    assert!(v["data_mismatch"].is_null(), "an unresolved source is not a mismatch:\n{v}");
    for field in ["data_unchecked", "data_change_unchecked"] {
        let note = v[field].as_str().unwrap_or_else(|| panic!("{field} names it:\n{v}"));
        assert!(note.contains("'a/fit.toml' stream 'cases'") && note.contains("0badf00d"),
            "{field}: {note}");
    }
}

/// Two `[data.holdout]` fits on identical training files, scored at the same
/// times, can still have scored different held-out counts. Only the holdout
/// files' recorded hashes show it.
#[test]
fn holdout_fits_whose_holdout_files_differ_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    two_patch_fit(dir, "a", data, 0.05,
        &[("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)]);
    let revised = TWO_PATCH_HOLDOUT.replace("42\turban\t11", "42\turban\t30");
    two_patch_fit(dir, "b", data, 0.05,
        &[("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", &revised)]);
    let out = compare(dir, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "different held-out data is refused:\n{stderr}");
    assert!(stderr.contains("different observed data") && stderr.contains("'cases' holdout")
            && !stderr.contains("'cases' ("),
        "names the holdout file, not the training file:\n{stderr}");
}

/// A `[data.holdout]` file edited after the fit ran is refused by the
/// changed-data check, like a training file.
#[test]
fn an_edited_holdout_file_is_refused_by_the_changed_data_check() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    let files = [("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)];
    let a = two_patch_fit(dir, "a", data, 0.05, &files);
    two_patch_fit(dir, "b", data, 0.06, &files);
    let v = json(&compare(dir, &["--format", "json"]));
    assert!(v["data_change_unchecked"].is_null(), "the holdout file was checked:\n{v}");

    let edited = TWO_PATCH_HOLDOUT.replace("42\turban\t11", "42\turban\t30");
    std::fs::write(a.join("holdout.tsv"), &edited).unwrap();
    let out = compare(dir, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    let current = sha256_hex(edited.as_bytes());
    for needle in ["changed after the fit ran", "'a/fit.toml' stream 'cases (holdout)'",
                   "holdout.tsv", &current[..8]] {
        assert!(stderr.contains(needle), "names {needle:?}:\n{stderr}");
    }
}

/// A sidecar written before full digests holds 8 characters. Its row is
/// compared on that prefix against a full record, never refused for the
/// length, and every format says the comparison was made on 8 characters.
#[test]
fn an_old_eight_character_sidecar_is_compared_on_its_prefix_and_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data.observations]\ncases = \"train.tsv\"\n";
    let a = two_patch_fit(dir, "a", data, 0.05, &[("train.tsv", TWO_PATCH_TRAIN)]);
    two_patch_fit(dir, "b", data, 0.06, &[("train.tsv", TWO_PATCH_TRAIN)]);
    let h = sha256_hex(TWO_PATCH_TRAIN.as_bytes());
    rewrite_sidecar(&a, "data_hashes", serde_json::json!({"cases": &h[..8]}));

    let v = json(&compare(dir, &["--format", "json"]));
    assert!(v["data_mismatch"].is_null() && v["data_changed"].is_null(), "{v}");
    let note = v["data_compared_on_prefix"].as_str()
        .unwrap_or_else(|| panic!("the JSON says so:\n{v}"));
    assert!(note.contains("'a/fit.toml'") && note.contains("8 characters")
            && !note.contains("'b/fit.toml'"), "{note}");
    for fmt in ["table", "md"] {
        let out = compare(dir, &["--format", fmt]);
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success() && text.contains("8 characters")
                && text.contains("'a/fit.toml'"), "{fmt}:\n{text}");
    }

    // A different file on the prefix is still a mismatch.
    rewrite_sidecar(&a, "data_hashes", serde_json::json!({"cases": "0badf00d"}));
    let out = compare(dir, &[]);
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
}

/// A `[data.holdout]` fit whose sidecar predates `holdout_hashes` scores its
/// holdout file with nothing recorded to check it against: both checks say
/// the holdout was not checked, and why.
#[test]
fn an_old_sidecar_without_holdout_hashes_reports_the_holdout_unchecked() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let data = "[data]\nfile = \"train.tsv\"\n\n[data.holdout]\ncases = \"holdout.tsv\"\n";
    let files = [("train.tsv", TWO_PATCH_TRAIN), ("holdout.tsv", TWO_PATCH_HOLDOUT)];
    let a = two_patch_fit(dir, "a", data, 0.05, &files);
    two_patch_fit(dir, "b", data, 0.06, &files);
    rewrite_sidecar(&a, "holdout_hashes", serde_json::Value::Null);

    let v = json(&compare(dir, &["--format", "json"]));
    assert!(v["data_mismatch"].is_null(), "an absent record is not a mismatch:\n{v}");
    let unchecked = v["data_unchecked"].as_str().unwrap_or_else(|| panic!("{v}"));
    assert!(unchecked.contains("'a/fit.toml' holdout 'cases'")
            && unchecked.contains("recorded no hash for its holdout file"), "{unchecked}");
    let unchanged = v["data_change_unchecked"].as_str().unwrap_or_else(|| panic!("{v}"));
    assert!(unchanged.contains("'a/fit.toml' stream 'cases (holdout)'")
            && unchanged.contains("holdout.tsv"), "{unchanged}");
}
