//! gh#636: `pfilter --save-prequential` / `--trace` accept NA holes.
//!
//! Multi-stream data whose streams start on different dates (leading NA
//! holes) is the NORMAL case; the old guard refused both outputs outright,
//! forcing fixed-θ diagnostics down to a single stream — which changes the
//! answer. Holes are now first-class: per-stream scores skip missing
//! (stream, time) cells, the joint covers present streams only, the trace
//! prints NA where nothing is present.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {}", bin.display());
    bin
}

#[test]
fn save_prequential_and_trace_accept_na_holes() {
    let bin = skip_if_missing_binary();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model = Path::new(&manifest).join("../../../ocaml/golden/sir_two_patch_long_obs.ir.json");
    let tmp = tempfile::tempdir().unwrap();
    // Rural starts one week late: a leading NA hole.
    let data = tmp.path().join("cases_long.tsv");
    std::fs::write(&data,
        "time\tpatch\tcases\n\
         7\turban\t12\n7\trural\tNA\n\
         14\turban\t18\n14\trural\t6\n\
         21\turban\t25\n21\trural\t9\n").unwrap();
    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.3\ngamma = 0.1\nrho = 0.6\nk = 5.0\n").unwrap();

    let stem = tmp.path().join("preq").to_string_lossy().into_owned();
    let trace = tmp.path().join("trace.tsv");
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &model.to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--data", &format!("cases={}", data.to_string_lossy()),
            "--save-prequential", &stem,
            "--trace", &trace.to_string_lossy(),
            "--particles", "50", "--dt", "1", "--seed", "1",
        ])
        .output()
        .expect("spawn pfilter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(),
        "holes must not refuse the diagnostic outputs (gh#636):\n{stderr}");
    assert!(stderr.contains("skip the missing"),
        "the hole handling is announced once:\n{stderr}");

    // The trace parses and the JSON round-trips (finite y_obs everywhere —
    // an all-hole step would have been omitted, not serialized as null).
    let json = std::fs::read_to_string(format!("{stem}.json")).unwrap();
    assert!(json.contains("per_stream"), "per-stream scores present");
    assert!(!json.contains("null"),
        "no non-finite score leaked into the JSON:\n{json}");
    let trace_txt = std::fs::read_to_string(&trace).unwrap();
    assert!(trace_txt.lines().count() > 3, "trace has rows:\n{trace_txt}");
}
