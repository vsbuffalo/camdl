//! `camdl batch` was removed
//! (`docs/dev/proposals/2026-10-24-batch-to-simulate.md` §4, §5): a
//! deterministic parameter grid is `simulate --sweep PARAM=GRID` /
//! `--sweep-file FILE`, and the store re-use, dry-run hit/miss view,
//! `--parallel` and `--force` of the retired verb live on `simulate`. The verb
//! is gone from the command tree rather than trapped, so a stale script fails
//! with clap's unrecognized-subcommand error (§6 test 12).

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

#[test]
fn batch_is_an_unrecognized_subcommand() {
    for sub in ["run", "status"] {
        let out = Command::new(camdl_bin())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .args(["batch", sub, "/no/such/sweep.toml"])
            .output()
            .expect("spawn camdl");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "clap's usage error exits 2; stderr: {stderr}");
        assert!(
            stderr.contains("unrecognized subcommand 'batch'"),
            "`camdl batch {sub}` must be clap's unrecognized-subcommand error, not a trap \
             or a run; got: {stderr}"
        );
    }
}
