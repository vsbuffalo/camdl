//! `--output-dir` named the results root camdl no longer has. Every command
//! refuses it as an unknown argument, and the error names the flags that
//! replace it, `--store DIR` for the run store's root and `-o` for an
//! export, in place of clap's nearest-spelling tip, `--output`, which on most
//! commands exports one table and names no store.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

#[test]
fn output_dir_is_refused_naming_the_store() {
    let bin = binary();
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let cases: [&[&str]; 6] = [
        &["fit", "run", "fit.toml", "--output-dir", "d"],
        &["simulate", "m.camdl", "--output-dir", "d"],
        &["pfilter", "m.camdl", "--output-dir=d"],
        &["profile", "--output-dir", "d"],
        &["predict", "@run", "--output-dir", "d"],
        &["list", "--output-dir", "d"],
    ];
    for args in cases {
        let out = Command::new(&bin).env_remove("CLICOLOR_FORCE").args(args).output().expect("spawn camdl");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?} must be a usage error:\n{stderr}");
        assert!(stderr.contains("unexpected argument '--output-dir'"), "{args:?}:\n{stderr}");
        assert!(stderr.contains("--store DIR"), "{args:?} must name `--store`:\n{stderr}");
        assert!(!stderr.contains("a similar argument exists"), "{args:?} must not suggest `--output`:\n{stderr}");
    }
}
