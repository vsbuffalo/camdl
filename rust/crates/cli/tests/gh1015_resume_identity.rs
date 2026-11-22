//! gh#1015. A resumed fit's identity is a function of what it resumes, not of
//! when its base ran.
//!
//! The `--resume` base is folded into the resumed run's `run_id`. Two runs of
//! the same fixed-seed base produce the same base run and the same chain
//! states, so resuming either must address the same resumed run: then the
//! same resumed fit run in two checkouts shares one cache entry. The base
//! run of `fixtures/reporting/pmmh_converged.toml`, and the resume of it by
//! `pmmh_resumed.toml`, are the ones `reporting_facts.rs` runs.

mod fit_store_support;

use std::path::PathBuf;
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

/// Run the base and the resume of it in a fresh copy of the fixture, each
/// work directory with a store of its own; return the base's and the resumed
/// fit's run ids.
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
            .env("CAMDL_STORE", work.join("store"))
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "camdl {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    };
    run(&["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    let base_run = fit_store_support::the_run(&work.join("store"));
    run(&["fit", "run", "pmmh_resumed.toml", "--seed", "1", "--progress", "none",
          "--resume", base_run.to_str().unwrap()]);
    let base_id = fit_store_support::run_id(&base_run);
    let runs = fit_store_support::runs(&work.join("store"));
    assert_eq!(runs.len(), 2, "the base and its resume: {runs:?}");
    let resumed = runs.iter().map(|r| fit_store_support::run_id(r)).find(|id| *id != base_id).unwrap();
    (base_id, resumed)
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
        "two runs of one fixed-seed base, each resumed alike, addressed different runs");
}
