//! Value choice (40 vs 80, NOT 80 vs 160). The number of output rows is set by
//! `output.times.regular.end` (80 in the golden), NOT by `simulation.t_end`,
//! so all of t_end ∈ {40,80,160} emit 81 rows ending at t=80. What t_end
//! changes is the trajectory VALUES: with t_end=40 the simulation stops
//! integrating at t=40 and the writer emits the frozen t=40 state for every
//! row t=41..80 (incidence columns zeroed), whereas t_end=80 runs the real
//! epidemic decline. So 40 vs 80 differ in bytes (verified by dumping the
//! tail), while 80 vs 160 are byte-identical (t_end >= output.end has no
//! effect). 40 vs 80 is the pair that exercises the cache-key gap; 80 vs 160
//! would pass for the wrong reason.
//!
//! gh#142 regression: the simulate cache key must include the `simulation`
//! block (`t_end`/`t_start`). Two models differing ONLY in `simulation.t_end`,
//! written to the SAME filename and the SAME `--cas` dir, must produce TWO
//! distinct CAS entries — not collide to one and serve the first's trajectory.
//!
//! Test-design note (the trap that caused a false negative while finding this):
//! the collision is invisible if the two models have DIFFERENT filenames,
//! because the path stem (`<model_stem>-<hash>`) then differs even when the
//! hash is identical. So both models MUST share a filename — then only the
//! hash can separate them, and a missing-input bug surfaces as one dir.

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

/// Write the golden IR with `simulation.t_end` set to `t_end`, to
/// `<dir>/model.ir.json` (same basename in every dir on purpose).
fn write_model_with_tend(dir: &Path, t_end: f64) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let raw = std::fs::read_to_string(golden_sir_basic()).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let model = v.get_mut("model").expect("enveloped IR has a `model` key");
    let sim = model.get_mut("simulation").expect("model has a `simulation` block");
    sim["t_end"] = serde_json::json!(t_end);
    let path = dir.join("model.ir.json");
    std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();
    path
}

#[test]
fn t_end_change_is_a_distinct_cache_entry() {
    let Some(bin) = skip_if_missing_binary() else { return; };
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("results");

    // Same filename (model.ir.json), different dirs, differ ONLY in t_end.
    let a = write_model_with_tend(&tmp.path().join("a"), 40.0);
    let b = write_model_with_tend(&tmp.path().join("b"), 80.0);

    let run = |model: &Path| {
        let st = Command::new(&bin)
            .args(["simulate", &model.to_string_lossy(),
                   "--scenario", "baseline", "--seed", "1", "--cas",
                   "--output-dir", &root.to_string_lossy(),
                   "-o", &tmp.path().join("traj.tsv").to_string_lossy()])
            .status().expect("spawn");
        assert!(st.success(), "simulate --cas should succeed");
    };
    run(&a);
    run(&b);

    let dirs = run_dirs(&root.join("sims"));
    assert_eq!(dirs.len(), 2,
        "two models differing only in simulation.t_end must produce TWO CAS \
         entries, not collide to one (gh#142). Found {} dir(s).", dirs.len());

    // And the recorded model_hash values must differ — the actual key gap.
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    let mut hashes: Vec<String> = dirs.iter().map(|d| {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("run.json")).unwrap()).unwrap();
        m["kind"]["model_hash"].as_str().unwrap().to_string()
    }).collect();
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), 2,
        "the two models must record distinct model_hash values (t_end is an input)");
    for h in &hashes {
        assert_ne!(h, EMPTY, "model_hash must not be the empty-input digest");
    }

    // And the trajectories themselves differ in bytes: t_end=40 freezes at its
    // t=40 state for t=41..80, t_end=80 runs the full decline. Identical bytes
    // here would mean one model was served the other's cached result (gh#142).
    let trajs: Vec<Vec<u8>> = dirs.iter()
        .map(|d| std::fs::read(d.join("traj.tsv")).unwrap()).collect();
    assert_ne!(trajs[0], trajs[1],
        "different t_end must yield different trajectories; identical bytes \
         means one model was served the other's cached result (gh#142)");
}
