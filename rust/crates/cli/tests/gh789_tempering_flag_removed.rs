//! gh#789: PGAS parallel tempering was removed. `--tempering` on `fit run` is
//! trapped and answered with the fix before any file is read, so a stale
//! script fails with the reason rather than clap's "unexpected argument".

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
fn tempering_flag_is_refused_with_the_removal_message() {
    let out = Command::new(camdl_bin())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", "/no/such/fit.toml", "--tempering", "1.0"])
        .output()
        .expect("spawn camdl");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "the trap exits 1; stderr: {stderr}");
    assert!(
        stderr.contains("--tempering") && stderr.contains("removed"),
        "the message must name the flag and say it was removed; got: {stderr}"
    );
}
