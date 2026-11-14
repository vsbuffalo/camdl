//! `camdl compare --streams` end to end: one two-stream model fitted twice,
//! once binding both of its observation sources and once binding one, and
//! the two fits compared on the stream they share.
//!
//! The model is the golden `seir_observations` IR: `weekly_cases` (a
//! negative-binomial count every 7 days) and `detection` (a Bernoulli reading
//! every 14 days), each its own source. A derived row is scored by a
//! `camdl pfilter` child, so each case runs two small IF2 fits.

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

fn golden_ir() -> PathBuf {
    manifest().join("../../../ocaml/golden/seir_observations.ir.json").canonicalize().unwrap()
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .current_dir(dir)
        .output().expect("spawn camdl")
}

const CASES: &str = "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n";
const DETECTION: &str = "time\tdetection\n14\t1\n28\t0\n";

/// Fit the model in `dir/name`, binding `sources` through
/// `[data.observations]`, with `holdout` written into `[data]`.
fn fit(dir: &Path, name: &str, sources: &[&str], holdout: &str) {
    let sub = dir.join(name);
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("cases.tsv"), CASES).unwrap();
    std::fs::write(sub.join("detection.tsv"), DETECTION).unwrap();
    let mut bound = String::new();
    for s in sources {
        let file = if *s == "weekly_cases" { "cases.tsv" } else { "detection.tsv" };
        bound.push_str(&format!("{s} = \"{file}\"\n"));
    }
    std::fs::write(sub.join("fit.toml"), format!(r#"
output_dir = "results"

[model]
camdl = "{ir}"

[data]
{holdout}
[data.observations]
{bound}
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
"#, ir = golden_ir().display())).unwrap();
    let out = camdl(&sub, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(out.status.success(), "fit run in {} failed:\n{}",
        sub.display(), String::from_utf8_lossy(&out.stderr));
}

fn compare(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["compare", "full/fit.toml", "cases_only/fit.toml",
                        "--particles", "100", "--replicates", "2"];
    args.extend_from_slice(extra);
    camdl(dir, &args)
}

/// Σ of one model's `log_score` for `stream` in a `.streams` pointwise file,
/// read from the candidate rows (`model`) and the baseline column
/// (`baseline`) alike.
fn pointwise_sum(tsv: &str, stream: &str, model: &str) -> f64 {
    let mut lines = tsv.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (m, b, s, ls, bls) = (col("model"), col("baseline"), col("stream"),
                              col("log_score"), col("baseline_log_score"));
    lines.map(|l| l.split('\t').collect::<Vec<_>>())
        .filter(|c| c[s] == stream)
        .filter_map(|c| {
            if c[m] == model { Some(c[ls].parse::<f64>().unwrap()) }
            else if c[b] == model { Some(c[bls].parse::<f64>().unwrap()) }
            else { None }
        })
        .sum()
}

/// Held out at the same boundary: the joint comparison is refused and names
/// `--streams` with the shared stream; with it, one table scores both fits on
/// `weekly_cases` alone, and each row's elpd is the sum of that stream's own
/// scores in the pointwise file.
#[test]
fn fits_of_different_stream_sets_compare_held_out_on_the_stream_they_share() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, "full", &["weekly_cases", "detection"], "holdout_after = 21.0");
    fit(dir, "cases_only", &["weekly_cases"], "holdout_after = 21.0");

    let out = compare(dir, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "the joint comparison is refused:\n{stderr}");
    assert!(stderr.contains("did not score the same streams")
            && stderr.contains("--streams weekly_cases"), "{stderr}");
    assert!(!stderr.contains("--allow-mismatched-horizon"), "{stderr}");

    let pw = dir.join("scores.tsv");
    let out = compare(dir, &["--streams", "weekly_cases", "--format", "json",
                             "--pointwise", pw.to_str().unwrap()]);
    assert!(out.status.success(), "compare --streams failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["streams_scored"], serde_json::json!(["weekly_cases"]));
    let tables = v["tables"].as_array().unwrap();
    assert_eq!(tables.len(), 1, "{v}");
    let t = &tables[0];
    assert_eq!(t["stream"], "weekly_cases");
    assert!(t["overrides"].as_array().unwrap().is_empty(), "held out, nothing waived: {t}");

    assert!(!pw.exists(), "no joint pointwise file under --streams");
    let streams_file = dir.join("scores.streams.tsv");
    let tsv = std::fs::read_to_string(&streams_file).unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no joint file was written"), "{stderr}");

    for row in t["rows"].as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        assert_eq!(row["conditioning"]["hold_out_tail"]["train_end"], serde_json::json!(21.0),
            "{row}");
        assert_eq!(row["t_score"], 2, "t = 28 and 35 lie past the boundary: {row}");
        let elpd = row["elpd"].as_f64().unwrap();
        let summed = pointwise_sum(&tsv, "weekly_cases", name);
        assert!((elpd - summed).abs() < 1e-8,
            "'{name}': elpd {elpd} is the sum of its weekly_cases scores {summed}\n{tsv}");
        let assimilated = if name.starts_with("full") {
            serde_json::json!(["detection", "weekly_cases"])
        } else {
            serde_json::json!(["weekly_cases"])
        };
        assert_eq!(row["streams_assimilated"], assimilated, "{row}");
        assert!(row["derivation"]["mc_se_elpd"].as_f64().is_some(),
            "two replicates give the stream's own filter-noise SE: {row}");
    }

    // The text table names its stream and the assimilated sets.
    let out = compare(dir, &["--streams", "weekly_cases"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let table = String::from_utf8_lossy(&out.stdout);
    assert!(table.starts_with("Stream: weekly_cases"), "{table}");
    assert!(table.contains("assimilated detection, weekly_cases")
            && table.contains("assimilated weekly_cases."), "{table}");

    // A stream one fit never bound cannot be scored for it.
    let out = compare(dir, &["--streams", "detection"]);
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("stream 'detection'") && stderr.contains("does not score 'detection'"),
        "{stderr}");
}

/// The same pair without a holdout is scored in-sample, where the ablation
/// is biased toward the fit trained on fewer streams: refused, and rendered
/// with that caveat under `--allow-in-sample-ablation`.
#[test]
fn an_in_sample_stream_ablation_is_refused_unless_waived() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, "full", &["weekly_cases", "detection"], "");
    fit(dir, "cases_only", &["weekly_cases"], "");

    let out = compare(dir, &["--streams", "weekly_cases"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("assimilated different streams") && stderr.contains("holdout_after")
            && stderr.contains("--allow-in-sample-ablation"), "{stderr}");

    let out = compare(dir, &["--streams", "weekly_cases", "--allow-in-sample-ablation"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let table = String::from_utf8_lossy(&out.stdout);
    let caveat: Vec<&str> = table.lines()
        .filter(|l| l.contains("--allow-in-sample-ablation")).collect();
    assert_eq!(caveat.len(), 1, "{table}");
    assert!(caveat[0].contains("leans toward the fit that assimilated fewer streams"), "{table}");

    let out = compare(dir, &["--streams", "weekly_cases", "--allow-in-sample-ablation",
                             "--format", "json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["tables"][0]["overrides"][0]["flag"], "--allow-in-sample-ablation", "{v}");
}
