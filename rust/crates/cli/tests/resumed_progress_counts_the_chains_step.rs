//! A resumed chain's progress reports the chain's own step.
//!
//! A chain resumed from step 2000 of 2500 is at step 2001 when it takes its
//! first step, and both the plain progress line and the `progress` event in
//! `events.jsonl` say so (`docs/camdl-run-spec.md` §10.11), as `progress.json`
//! does. Counting from 1 instead would show `chain 0 1/2500` for a chain
//! four-fifths done, and an ETA for 2500 steps where 500 remain. The base and
//! the resume are the reporting fixture's `pmmh_converged.toml` and
//! `pmmh_resumed.toml`.

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

fn leaf(work: &Path, fit: &str) -> PathBuf {
    only_child(&only_child(&only_child(&work.join("results/fits"), &format!("{fit}-")), "pmmh-"), "seed_")
}

#[test]
fn a_resumed_chain_reports_its_own_step() {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_resumed_progress_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for f in ["sir.camdl", "cases.tsv", "pmmh_converged.toml", "pmmh_resumed.toml"] {
        std::fs::copy(manifest().join("tests/fixtures/reporting").join(f), work.join(f)).unwrap();
    }
    let run = |args: &[&str]| -> String {
        let out = Command::new(&camdl)
            .env_remove("CLICOLOR_FORCE")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            .env("CAMDL_IR_CACHE_DIR", work.join(".ir-cache"))
            .current_dir(&work)
            .args(args)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "camdl {args:?}: {stderr}");
        stderr
    };
    run(&["fit", "run", "pmmh_converged.toml", "--seed", "1", "--progress", "none"]);
    let base = leaf(&work, "pmmh_converged");
    let stderr = run(&["fit", "run", "pmmh_resumed.toml", "--seed", "1", "--progress", "plain",
        "--resume", base.to_str().unwrap()]);

    // The plain line: `chain 0 2001/2500  ll=…`, the first one a chain says.
    for chain in 0..4 {
        let label = format!("chain {chain} ");
        let first = stderr.lines()
            .filter_map(|l| l.trim().strip_prefix(label.as_str()))
            .find_map(|rest| rest.split_whitespace().next().and_then(|c| c.split_once('/')))
            .unwrap_or_else(|| panic!("no plain progress line for chain {chain}:\n{stderr}"));
        assert_eq!(first, ("2001", "2500"), "chain {chain}'s first progress line");
    }

    // The events: each chain's first `progress` is its step 2001 of 2500.
    let log = std::fs::read_to_string(leaf(&work, "pmmh_resumed").join("events.jsonl")).unwrap();
    let mut firsts = std::collections::BTreeMap::new();
    for line in log.lines() {
        let e: serde_json::Value = serde_json::from_str(line).unwrap();
        if e["event"] == "progress" {
            firsts.entry(e["chain"].as_u64().unwrap())
                .or_insert((e["step"].as_u64().unwrap(), e["of"].as_u64().unwrap()));
        }
    }
    let expected: std::collections::BTreeMap<u64, (u64, u64)> =
        (0..4).map(|c| (c, (2001, 2500))).collect();
    assert_eq!(firsts, expected, "each chain's first progress event (step, of)");
}
