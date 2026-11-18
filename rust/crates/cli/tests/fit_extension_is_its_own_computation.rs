//! A fit extended to a longer target is a different computation, and the
//! shorter fit is left exactly as it was.
//!
//! PMMH and PGAS can be extended (`mh` cannot: `fit run` refuses `--resume`
//! for every other method): `fit run --resume <base>` with a
//! longer `iterations` / `sweeps` continues each chain from the state the base
//! recorded at its end (`chain_N/resume_state.bin`, the checkpoint). Four
//! things are kept distinct:
//!
//! - the computation's identity, which includes its target length: the
//!   extended fit has its own run id;
//! - the checkpoint, a state a longer computation can start from, which is
//!   never served as a completed result: it is admitted only by the
//!   extension-free key it records (`cas::resume_guard`), and only toward a
//!   longer target;
//! - the completed result at a target length: an ordinary leaf, the base's
//!   unchanged by any extension of it;
//! - the executions that produce or extend them, recorded per leaf outside its
//!   stored bytes.
//!
//! Fixture: `tests/fixtures/reporting/sir.camdl` with `cases.tsv`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Work(PathBuf);
impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn work(tag: &str, configs: &[(&str, String)]) -> Work {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("camdl_extension_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["sir.camdl", "cases.tsv"] {
        std::fs::copy(manifest_dir().join("tests/fixtures/reporting").join(f), dir.join(f)).unwrap();
    }
    for (name, body) in configs {
        std::fs::write(dir.join(name), body).unwrap();
    }
    Work(dir)
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    let bin = manifest_dir().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let camdlc = manifest_dir().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    Command::new(bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc)
        .env_remove("CAMDL_OUTPUT_DIR")
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap()
}

fn ok(out: Output, what: &str) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{what}: {stderr}");
    stderr
}

/// The `fit_method` leaves under the work dir's store, by their run id.
fn leaves(dir: &Path) -> BTreeMap<String, PathBuf> {
    fn walk(d: &Path, out: &mut BTreeMap<String, PathBuf>) {
        let Ok(entries) = std::fs::read_dir(d) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name().is_some_and(|n| n == ".attempts") {
                continue;
            }
            if p.is_dir() {
                walk(&p, out);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let rec: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                if rec["kind"] == "fit_method" {
                    out.insert(rec["run_id"].as_str().unwrap().to_string(), p.parent().unwrap().to_path_buf());
                }
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(&dir.join("results/fits"), &mut out);
    out
}

fn record(leaf: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap()
}

/// Every file under `leaf`, by relative path, with its bytes: the whole
/// leaf, records of the attempt included, so "unchanged" means unchanged.
fn snapshot(leaf: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, d: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(root, &p, out);
            } else {
                out.insert(p.strip_prefix(root).unwrap().display().to_string(), std::fs::read(&p).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(leaf, leaf, &mut out);
    out
}

/// The largest index (`step` or `sweep`, the first column) in a chain's trace.
fn last_index(leaf: &Path) -> usize {
    std::fs::read_to_string(leaf.join("chain_0/trace.tsv")).unwrap()
        .lines().skip(1).filter_map(|l| l.split('\t').next()?.parse().ok()).max().unwrap()
}

const HEAD: &str = r#"[model]
camdl = "sir.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
beta  = { start = 0.5 }
gamma = { start = 0.2 }

[fixed]
rho = 0.5
N0  = 2000
"#;

fn pmmh(iterations: usize) -> String {
    format!("{HEAD}\n[method]\nalgorithm  = \"pmmh\"\nbackend    = \"chain_binomial\"\nchains     = 2\n\
             particles  = 50\niterations = {iterations}\nburn_in    = 100\nthin       = 10\n")
}

fn pgas(sweeps: usize) -> String {
    format!("{HEAD}\n[method]\nalgorithm = \"pgas\"\nbackend   = \"chain_binomial\"\nchains    = 2\n\
             particles = 50\nsweeps    = {sweeps}\nburn_in   = 100\n")
}

/// Fit `base.toml`, extend it with `--resume` to `longer.toml`'s target, and
/// check what the extension did and did not touch.
fn extend_and_check(tag: &str, base_toml: String, longer_toml: String, base_len: usize, longer_len: usize) {
    let w = work(tag, &[("base.toml", base_toml), ("longer.toml", longer_toml)]);
    let fit = ["--seed", "1", "--progress", "none"];
    ok(camdl(&w.0, &[&["fit", "run", "base.toml"][..], &fit].concat()), "the base fit");
    let before = leaves(&w.0);
    assert_eq!(before.len(), 1);
    let (base_id, base_leaf) = before.into_iter().next().unwrap();
    let base_bytes = snapshot(&base_leaf);
    let summary = |leaf: &Path| ok(camdl(&w.0, &["fit", "summary", &leaf.to_string_lossy()]), "fit summary");
    let base_summary = summary(&base_leaf);
    assert!(last_index(&base_leaf) < base_len);

    ok(camdl(&w.0, &[&["fit", "run", "longer.toml", "--resume", &base_leaf.to_string_lossy()][..], &fit].concat()),
        "the extension");
    let after = leaves(&w.0);
    assert_eq!(after.len(), 2, "the extension is a leaf of its own: {after:?}");

    // The shorter fit's leaf is unchanged, byte for byte, records of its
    // attempt included.
    assert_eq!(snapshot(&base_leaf), base_bytes, "extending the fit changed its base's leaf");
    // It is still found under its own run id, which it still records.
    assert_eq!(after.get(&base_id), Some(&base_leaf));
    assert_eq!(record(&base_leaf)["run_id"], base_id.as_str());
    let (long_id, long_leaf) = after.iter().find(|(id, _)| **id != base_id).unwrap();
    assert_ne!(long_leaf, &base_leaf);
    assert_eq!(record(long_leaf)["inputs"]["resume"]["parent_run_id"], base_id.as_str());
    assert!(record(&base_leaf)["inputs"].get("resume").is_none());
    // It is not reported as the longer one: its own chain still ends below its
    // own target, its summary reads as it did, and the extension's runs on.
    assert!(last_index(&base_leaf) < base_len);
    assert!(last_index(long_leaf) >= base_len && last_index(long_leaf) < longer_len);
    assert_eq!(summary(&base_leaf), base_summary, "the base's summary changed after the extension");
    // Asked for again, the shorter fit is the stored base.
    let again = ok(camdl(&w.0, &[&["fit", "run", "base.toml"][..], &fit].concat()), "the base again");
    assert!(again.contains("cache hit"), "the base was recomputed rather than found: {again}");
    assert_eq!(leaves(&w.0).len(), 2);
    assert_ne!(long_id, &base_id);
}

#[test]
fn extending_a_pmmh_fit_is_a_new_computation_and_leaves_the_shorter_fit_intact() {
    extend_and_check("pmmh", pmmh(300), pmmh(500), 300, 500);
}

#[test]
fn extending_a_pgas_fit_is_a_new_computation_and_leaves_the_shorter_fit_intact() {
    extend_and_check("pgas", pgas(200), pgas(300), 200, 300);
}

/// A checkpoint is never served as a completed result at a target it does
/// not match. Resuming toward a target the base has already reached would
/// address a leaf for that target holding the base's whole chain — a
/// 300-step chain filed as the 200-step computation — so it is refused before
/// any leaf is claimed; at the base's own target, the base is the result.
#[test]
fn resuming_to_a_target_the_base_already_reached_is_refused() {
    for (tag, base, other, len, unit) in [
        ("pmmh", pmmh(300), [pmmh(200), pmmh(300)], 300, "steps"),
        ("pgas", pgas(200), [pgas(150), pgas(200)], 200, "sweeps"),
    ] {
        let [shorter, same] = other;
        let w = work(&format!("refuse_{tag}"), &[("base.toml", base), ("shorter.toml", shorter), ("same.toml", same)]);
        let fit = ["--seed", "1", "--progress", "none"];
        ok(camdl(&w.0, &[&["fit", "run", "base.toml"][..], &fit].concat()), "the base fit");
        let (_, base_leaf) = leaves(&w.0).into_iter().next().unwrap();
        for cfg in ["shorter.toml", "same.toml"] {
            let out = camdl(&w.0, &[&["fit", "run", cfg, "--resume", &base_leaf.to_string_lossy()][..], &fit].concat());
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(!out.status.success(), "{tag} {cfg}: a resume toward a reached target ran:\n{stderr}");
            assert!(stderr.contains(&format!("already holds {len} {unit}")), "{tag} {cfg}: {stderr}");
            assert_eq!(leaves(&w.0).len(), 1, "{tag} {cfg}: a leaf was written for the refused resume");
        }
    }
}
