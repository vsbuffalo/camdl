//! `lineage tree`, `sojourn` and `cohort` fail when their `--output` file
//! cannot be written. They discarded every write and flush error (`.ok()`),
//! so a write that failed part-way exited 0 and left a truncated file that
//! read as the whole result.
//!
//! The failure is made real with the file-size limit: the child runs with
//! `RLIMIT_FSIZE` far below what each command writes and with `SIGXFSZ`
//! ignored, so an over-limit write returns `EFBIG` instead of killing it.
#![cfg(unix)]

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn camdl(bin: &Path, dir: &Path, args: &[&str]) -> Command {
    let mut c = Command::new(bin);
    c.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
        .current_dir(dir)
        .args(args);
    c
}

/// Bytes a limited child may write to any one file: less than the header
/// line of every table here, so the first flush fails.
const FILE_SIZE_LIMIT: libc::rlim_t = 8;

#[test]
fn a_failed_output_write_is_an_error() {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    let tmp = std::env::temp_dir().join(format!("camdl_lineage_write_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(manifest().join("tests/fixtures/reporting/lineage.camdl"), tmp.join("lineage.camdl"))
        .unwrap();
    for args in [
        &[
            "simulate", "lineage.camdl", "--param", "beta=0.8", "--param", "gamma=0.1", "--seed", "3",
            "--event-log", "ev.tsv", "--tsv", "--backend", "gillespie",
        ][..],
        &["lineage", "realize", "ev.tsv", "--identity-seed", "7", "-o", "ll.tsv"][..],
    ] {
        let out = camdl(&bin, &tmp, args).output().expect("spawn camdl");
        assert!(out.status.success(), "{args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
    }

    let cases: [(&str, &[&str]); 3] = [
        ("tree.nwk", &["lineage", "tree", "ll.tsv", "--scheme", "flat:1.0", "--output", "tree.nwk"]),
        ("soj.tsv", &["lineage", "sojourn", "ll.tsv", "--compartment", "1", "--output", "soj.tsv"]),
        ("coh.tsv", &["lineage", "cohort", "ll.tsv", "--event", "infection", "--window", "7", "--output", "coh.tsv"]),
    ];
    for (file, args) in cases {
        let mut cmd = camdl(&bin, &tmp, args);
        // SAFETY: between fork and exec only async-signal-safe calls are
        // made (`signal`, `setrlimit`), touching nothing the parent shares.
        unsafe {
            cmd.pre_exec(|| {
                libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
                let lim = libc::rlimit { rlim_cur: FILE_SIZE_LIMIT, rlim_max: FILE_SIZE_LIMIT };
                if libc::setrlimit(libc::RLIMIT_FSIZE, &lim) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let out = cmd.output().expect("spawn camdl");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "{args:?}: the {file} write failed, so the command must fail:\n{stderr}"
        );
        assert!(stderr.contains(file), "{args:?}: the error names {file}:\n{stderr}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
