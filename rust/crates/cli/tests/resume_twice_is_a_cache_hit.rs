//! A second identical `fit run --resume` is a cache hit.
//!
//! A resumed run's identity is its file's, its seed, the longer chain's
//! length and a dependency on the base leaf's content (gh#1015), so running
//! the same resume again names the same inputs, and the store's rule is that
//! the same inputs are the same leaf (`docs/camdl-run-spec.md` §3.7): the
//! completed leaf is reused, not claimed again. The base and the resume are
//! the reporting fixture's `pmmh_converged.toml` and `pmmh_resumed.toml`.

mod fit_store_support;

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

/// The one stored fit run of the fit file `fit` (`<fit>.toml` in `work`):
/// the run in the store at `work/camdl-store` whose `fit.toml`, the file as
/// given, is that file's bytes.
fn leaf(work: &Path, fit: &str) -> PathBuf {
    let text = std::fs::read(work.join(format!("{fit}.toml"))).unwrap();
    let hits: Vec<PathBuf> = fit_store_support::runs(&work.join("camdl-store"))
        .into_iter()
        .filter(|r| std::fs::read(r.join("fit.toml")).ok().as_deref() == Some(&text[..]))
        .collect();
    assert_eq!(hits.len(), 1, "expected one fit run of {fit}.toml: {hits:?}");
    hits.into_iter().next().unwrap()
}

#[test]
fn resuming_the_same_base_again_reuses_the_resumed_leaf() {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_resume_twice_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for f in ["sir.camdl", "cases.tsv", "pmmh_converged.toml", "pmmh_resumed.toml"] {
        std::fs::copy(manifest().join("tests/fixtures/reporting").join(f), work.join(f)).unwrap();
    }
    let run = |args: &[&str]| -> (bool, String) {
        let out = Command::new(&camdl)
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            .env("CAMDL_IR_CACHE_DIR", work.join(".ir-cache"))
            .env("CAMDL_STORE", work.join("camdl-store"))
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    let (ok, stderr) = run(&["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    assert!(ok, "base: {stderr}");
    let base = leaf(&work, "pmmh_converged");
    let resume = ["fit", "run", "pmmh_resumed.toml", "--seed", "1", "--progress", "none",
        "--resume", base.to_str().unwrap()];
    let (ok, stderr) = run(&resume);
    assert!(ok, "first resume: {stderr}");
    let resumed = leaf(&work, "pmmh_resumed");
    let log = std::fs::read(resumed.join("events.jsonl")).unwrap();

    let (ok, stderr) = run(&resume);
    assert!(ok, "the second, identical resume must succeed:\n{stderr}");
    // Named relative to the store root, as every cache hit is.
    let rel = resumed.strip_prefix(work.join("camdl-store")).unwrap().display().to_string();
    assert!(stderr.lines().any(|l| l.contains("cache hit — reusing") && l.contains(&rel)),
        "the second resume reuses {rel}:\n{stderr}");
    assert_eq!(leaf(&work, "pmmh_resumed"), resumed, "one resumed run");
    assert_eq!(std::fs::read(resumed.join("events.jsonl")).unwrap(), log,
        "a cache hit runs no method, so the stored leaf's events.jsonl is unchanged");
}
