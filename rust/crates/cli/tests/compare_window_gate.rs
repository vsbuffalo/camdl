//! gh#833, phase 3: `compare` refuses to difference two traces that scored the
//! same stream at the same times over DIFFERENT windows.
//!
//! The case a missing row used to hide. A file with rows closing at
//! …, 9, 10, 12, 13, … can mean two things: the day (10, 11) was never
//! observed (a gap — row `[11, 12)` stands alone), or the row closing at 12 is
//! a two-day total (a merge — `[10, 12)`). Under row-spacing inference both
//! read as the merge. With per-row window columns each reading is a statement,
//! the two fits score at identical times, and only their likelihoods differ —
//! which `compare` would otherwise report as a model-comparison result.
//!
//! Exercised through the real path: `camdl pfilter --save-prequential` on each
//! reading, then `camdl compare` on the two `prequential.json` files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn seed_timing_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json")
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_wgate_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// The fixture with its time column replaced by a `window_start`/`window_stop`
/// pair and `covers` set to `window_columns` — the per-row form, the only one
/// that can state a gap or a merge.
fn windowed_model(dir: &Path) -> PathBuf {
    let src = std::fs::read_to_string(seed_timing_ir()).unwrap();
    let with_roles = src.replacen(
        "{ \"name\": \"time\", \"role\": \"time\" },",
        "{ \"name\": \"win_start\", \"role\": \"window_start\" },\n          \
         { \"name\": \"win_stop\", \"role\": \"window_stop\" },",
        1,
    );
    assert!(with_roles.contains("window_stop"), "window role injection failed");
    let injected = with_roles.replacen(
        "\"projection\":",
        "\"covers\":{\"kind\":\"window_columns\"},\"projection\":",
        1,
    );
    assert!(injected.contains("window_columns"), "covers injection failed");
    let p = dir.join("windowed.ir.json");
    std::fs::write(&p, injected).unwrap();
    p
}

/// Daily counts for D in 3..43 with a bump so the series is not all zeros.
fn daily_count(k: u32) -> u32 {
    if k >= 20 { (k - 19) * 3 } else { 0 }
}

/// Windows `[D, D+1)`, the row for `[10, 11)` OMITTED: a gap.
fn write_gapped(path: &Path) {
    let mut s = String::from("win_start\twin_stop\tcases\n");
    for k in 0..40u32 {
        let d = 3 + k;
        if d == 10 {
            continue;
        }
        s.push_str(&format!("{d}\t{}\t{}\n", d + 1, daily_count(k)));
    }
    std::fs::write(path, s).unwrap();
}

/// The same stops, but the row closing at 12 covers `[10, 12)`: a merge. Its
/// count is the two days' total, which is what a merged row would carry.
fn write_merged(path: &Path) {
    let mut s = String::from("win_start\twin_stop\tcases\n");
    for k in 0..40u32 {
        let d = 3 + k;
        if d == 10 {
            continue;
        }
        if d == 11 {
            s.push_str(&format!("10\t12\t{}\n", daily_count(k - 1) + daily_count(k)));
            continue;
        }
        s.push_str(&format!("{d}\t{}\t{}\n", d + 1, daily_count(k)));
    }
    std::fs::write(path, s).unwrap();
}

const BASE_PARAMS: &[&str] = &[
    "--param", "beta=0.6",
    "--param", "gamma=0.2",
    "--param", "lambda=2.0",
    "--param", "w=3.0",
    "--param", "N0=5000",
    "--param", "rho=0.5",
    "--param", "k=20",
    "--param", "tau=2",
];

/// Filter `data` under `model` and save the prequential trace; returns the
/// `.json` path.
fn save_trace(camdl: &Path, model: &Path, data: &Path, stem: &Path, seed: &str) -> PathBuf {
    let stem_s = stem.to_str().unwrap();
    let mut args = vec![
        "pfilter", model.to_str().unwrap(),
        "--particles", "200", "--dt", "1", "--seed", seed,
        "--data", data.to_str().unwrap(),
        "--save-prequential", stem_s,
    ];
    args.extend_from_slice(BASE_PARAMS);
    let out = run(camdl, &args);
    assert!(
        out.status.success(),
        "pfilter --save-prequential failed:\nSTDOUT:{}\nSTDERR:{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let json = PathBuf::from(format!("{stem_s}.json"));
    assert!(json.exists(), "no trace written at {}", json.display());
    json
}

#[test]
fn compare_refuses_a_gapped_reading_against_a_merged_one() {
    let camdl = camdl_bin();
    let tmp = tempdir("gap_vs_merge");
    let model = windowed_model(&tmp);

    let gapped_tsv = tmp.join("gapped.tsv");
    let merged_tsv = tmp.join("merged.tsv");
    write_gapped(&gapped_tsv);
    write_merged(&merged_tsv);

    // Stems distinct from the data files: `--save-prequential STEM` writes
    // `STEM.tsv` beside `STEM.json`.
    let gapped = save_trace(&camdl, &model, &gapped_tsv, &tmp.join("preq_gapped"), "5");
    let merged = save_trace(&camdl, &model, &merged_tsv, &tmp.join("preq_merged"), "5");

    // The trace itself records what each value covered, in the tagged form:
    // the row closing at 12 covers [11, 12) in the gapped reading.
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&gapped).unwrap()).unwrap();
    assert_eq!(trace["schema_version"], 4);
    let at_12 = trace["steps"].as_array().unwrap().iter()
        .find(|s| s["t"] == 12.0)
        .expect("a scored step at t=12");
    assert_eq!(
        at_12["per_stream"][0]["coverage"],
        serde_json::json!({"kind": "interval", "start": 11.0, "stop": 12.0}),
        "the saved trace must carry per-stream coverage: {at_12}"
    );

    let out = run(&camdl, &[
        "compare", gapped.to_str().unwrap(), merged.to_str().unwrap(),
        "--baseline", "preq_gapped.json",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(), Some(2),
        "compare must refuse (exit 2) a gapped reading against a merged one:\nSTDOUT:{}\nSTDERR:{stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(stderr.contains("different windows"), "names the defect: {stderr}");
    assert!(stderr.contains("[11, 12)") && stderr.contains("[10, 12)"),
        "names both windows: {stderr}");
    assert!(stderr.contains("t=12"), "and the step: {stderr}");

    // Positive control: the same reading twice (a different filter seed, so
    // the scores differ) is a fair comparison and renders.
    let gapped_again = save_trace(&camdl, &model, &gapped_tsv, &tmp.join("preq_gapped2"), "6");
    let out = run(&camdl, &[
        "compare", gapped.to_str().unwrap(), gapped_again.to_str().unwrap(),
        "--baseline", "preq_gapped.json",
    ]);
    assert!(
        out.status.success(),
        "identical windows must compare:\nSTDOUT:{}\nSTDERR:{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
