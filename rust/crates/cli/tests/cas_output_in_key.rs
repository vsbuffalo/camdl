//! gh#142 sibling: the simulate cache key must include the `output` block, not
//! just `simulation`. `output.times.regular.end` governs the emitted row count
//! and horizon, so two models differing ONLY in `output.end` produce genuinely
//! different trajectories (40 → 41 data rows ending at t=40; 80 → 81 rows
//! ending at t=80). Written to the SAME filename and SAME `--cas` dir, they
//! must produce TWO distinct CAS entries — not collide to one and serve the
//! first's (shorter) trajectory.
//!
//! This is the EXACT collision class as `cas_tend_in_key.rs`, one block over:
//! `model_hash` hashed an allowlist of structural keys that included
//! `simulation` (the gh#142 fix) but not `output` / `model_structure` /
//! `bindings` — all read at runtime. The allowlist IS the hand-written field
//! list the total-input-hash proposal argues against; this test pins the
//! `output` member of it. 40 vs 80 differ in row count, so the trajectory
//! comparison can't pass for the wrong reason.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> Option<PathBuf> {
    let bin = binary();
    if !bin.exists() {
        eprintln!("skipping: camdl binary not built at {}", bin.display());
        return None;
    }
    Some(bin)
}

fn golden_sir_basic() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/sir_basic.ir.json")
}

/// Collect every dir under `root` containing a `run.json`.
fn run_dirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut has = false;
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); }
            else if p.file_name().and_then(|n| n.to_str()) == Some("run.json") { has = true; }
        }
        if has { out.push(d); }
    }
    out
}

/// Write the golden IR with `output.times.regular.end` set to `end`, to
/// `<dir>/model.ir.json` (same basename in every dir on purpose).
fn write_model_with_output_end(dir: &Path, end: f64) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let raw = std::fs::read_to_string(golden_sir_basic()).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let model = v.get_mut("model").expect("enveloped IR has a `model` key");
    let out = model.get_mut("output").expect("model has an `output` block");
    out["times"]["regular"]["end"] = serde_json::json!(end);
    let path = dir.join("model.ir.json");
    std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();
    path
}

#[test]
fn output_end_change_is_a_distinct_cache_entry() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("results");

    // Same filename (model.ir.json), different dirs, differ ONLY in output.end.
    let a = write_model_with_output_end(&tmp.path().join("a"), 40.0);
    let b = write_model_with_output_end(&tmp.path().join("b"), 80.0);

    let run = |model: &Path, out: &Path| {
        let st = Command::new(&bin)
            .args(["simulate", &model.to_string_lossy(),
                   "--scenario", "baseline", "--seed", "1", "--cas",
                   "--output-dir", &root.to_string_lossy(),
                   "-o", &out.to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success(), "simulate --cas should succeed");
    };
    let a_out = tmp.path().join("a.tsv");
    let b_out = tmp.path().join("b.tsv");
    run(&a, &a_out);
    run(&b, &b_out);

    let dirs = run_dirs(&root.join("sims"));
    assert_eq!(dirs.len(), 2,
        "two models differing only in output.times.regular.end must produce \
         TWO CAS entries, not collide to one (gh#142 class). Found {} dir(s).",
        dirs.len());

    // The recorded model_hash values must differ — the actual key gap.
    let mut hashes: Vec<String> = dirs.iter().map(|d| {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("run.json")).unwrap()).unwrap();
        m["kind"]["model_hash"].as_str().unwrap().to_string()
    }).collect();
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), 2,
        "the two models must record distinct model_hash values (output is an input)");

    // The output bytes themselves differ: end=40 emits 41 data rows ending at
    // t=40, end=80 emits 81 ending at t=80. Identical bytes here would mean one
    // model was served the other's cached result (the gh#142-class bug).
    let a_bytes = std::fs::read(&a_out).unwrap();
    let b_bytes = std::fs::read(&b_out).unwrap();
    assert_ne!(a_bytes, b_bytes,
        "different output.end must yield different trajectories; identical \
         bytes means one model was served the other's cached result");
    // And concretely: A ends at t=40, B at t=80.
    let last_t = |bytes: &[u8]| -> String {
        let s = String::from_utf8_lossy(bytes);
        let last = s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("");
        last.split('\t').next().unwrap_or("").to_string()
    };
    assert_eq!(last_t(&a_bytes), "40", "A (output.end=40) should end at t=40");
    assert_eq!(last_t(&b_bytes), "80", "B (output.end=80) should end at t=80");
}
