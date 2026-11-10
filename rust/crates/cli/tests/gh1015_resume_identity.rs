//! gh#1015. A resumed fit's identity is a function of what it resumes, not of
//! when its base ran.
//!
//! The `--resume` base is folded into the resumed leaf's `run_id` as a dep.
//! Two runs of the same fixed-seed base produce the same base leaf and the same
//! chain states, so resuming either must address the same resumed leaf: then
//! the same resumed fit run in two checkouts shares one cache entry. The base
//! run of `fixtures/reporting/pmmh_converged.toml`, and the resume of it by
//! `pmmh_resumed.toml`, are the ones `reporting_facts.rs` runs.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The single directory under `dir` whose name starts with `prefix`.
fn only_child(dir: &Path, prefix: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    assert_eq!(hits.len(), 1, "expected one `{prefix}*` under {}: {hits:?}", dir.display());
    hits.into_iter().next().unwrap()
}

/// The method directory `results/fits/<fit>-*/pmmh-*`, whose name carries
/// the method level's hash.
fn method_dir(work: &Path, fit: &str) -> PathBuf {
    only_child(&only_child(&work.join("results/fits"), &format!("{fit}-")), "pmmh-")
}

/// Run the base and the resume of it in a fresh copy of the fixture; return
/// the base's and the resumed fit's method directory names.
fn base_then_resume(tag: &str) -> (String, String) {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_gh1015_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for f in ["sir.camdl", "cases.tsv", "pmmh_converged.toml", "pmmh_resumed.toml"] {
        std::fs::copy(manifest().join("tests/fixtures/reporting").join(f), work.join(f)).unwrap();
    }
    let run = |args: &[&str]| {
        let out = Command::new(&camdl)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "camdl {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    };
    run(&["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    let base_dir = method_dir(&work, "pmmh_converged");
    let base_leaf = only_child(&base_dir, "seed_");
    run(&["fit", "run", "pmmh_resumed.toml", "--seed", "1", "--progress", "none",
          "--resume", base_leaf.to_str().unwrap()]);
    let name = |p: PathBuf| p.file_name().unwrap().to_string_lossy().into_owned();
    (name(base_dir), name(method_dir(&work, "pmmh_resumed")))
}

#[test]
fn resuming_two_runs_of_one_base_addresses_one_leaf() {
    let (base_a, resumed_a) = base_then_resume("a");
    // The base records when it ran to the second; a second apart, two runs of
    // it differ in that alone.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let (base_b, resumed_b) = base_then_resume("b");
    assert_eq!(base_a, base_b, "the base itself must not re-key between runs");
    assert_eq!(resumed_a, resumed_b,
        "two runs of one fixed-seed base, each resumed alike, addressed different leaves");
}
