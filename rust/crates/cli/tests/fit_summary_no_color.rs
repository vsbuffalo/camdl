//! `fit summary --no-color` writes no colour to either stream.
//!
//! The flag outranks every environment variable (`Painter::for_stream`), so
//! under `CLICOLOR_FORCE=1` it still means no escape codes — on stdout, where
//! the summary goes, and on stderr, where its warnings and notes go.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let p = manifest().join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build-rust` (gh#105)", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn only_fit(work: &Path, prefix: &str) -> PathBuf {
    let hits: Vec<PathBuf> = std::fs::read_dir(work.join("results/fits")).unwrap().flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    assert_eq!(hits.len(), 1, "{hits:?}");
    hits.into_iter().next().unwrap()
}

#[test]
fn no_color_holds_on_stderr_under_clicolor_force() {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let work = std::env::temp_dir().join(format!("camdl_summary_no_color_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    let _guard = Tmp(work.clone());
    for e in std::fs::read_dir(manifest().join("tests/fixtures/reporting")).unwrap().flatten() {
        std::fs::copy(e.path(), work.join(e.file_name())).unwrap();
    }
    let camdl = |args: &[&str], force: bool| {
        let mut c = Command::new(camdl());
        c.env_remove("CLICOLOR_FORCE").env_remove("NO_COLOR")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc())
            .current_dir(&work)
            .args(args);
        if force {
            c.env("CLICOLOR_FORCE", "1");
        }
        let out = c.output().unwrap();
        assert!(out.status.success(), "{args:?}:\n{}", String::from_utf8_lossy(&out.stderr));
        (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    camdl(&["fit", "run", "pmmh_stuck.toml", "--seed", "8", "--progress", "none"], false);
    let fit = only_fit(&work, "pmmh_stuck-");
    // `--exclude-chains 2` drops the outlier chain, so the summary's stderr
    // carries both the identifiability note and the exclusion warning.
    let (stdout, stderr) = camdl(
        &["fit", "summary", fit.to_str().unwrap(), "--exclude-chains", "2", "--no-color"], true);
    let esc = char::from(0x1b);
    assert!(stderr.contains("warning: --exclude-chains dropped chain(s) 2"), "{stderr}");
    assert!(stderr.contains("note: chain 2 disagree strongly"), "{stderr}");
    assert!(!stdout.contains(esc), "stdout carries colour under --no-color:\n{stdout:?}");
    assert!(!stderr.contains(esc), "stderr carries colour under --no-color:\n{stderr:?}");

    // Without the flag, CLICOLOR_FORCE colours both streams: the test above
    // is not passing because nothing is ever coloured.
    let (stdout, stderr) = camdl(
        &["fit", "summary", fit.to_str().unwrap(), "--exclude-chains", "2"], true);
    assert!(stdout.contains(esc) && stderr.contains(esc), "CLICOLOR_FORCE colours both streams");
}
