//! L9 (external validation) gate for `cargo test`.
//!
//! Shells out to the `external-harness` binary's `run-all` mode
//! (fast-path: cached fixtures only, no R/Python/Stan required). If
//! any case fails — tolerance breach, stale fixture, camdl crash —
//! this test fails and the harness's stderr is surfaced in cargo's
//! test output.
//!
//! Rationale for shell-out rather than linking the harness as a
//! library: the harness spawns camdl subprocesses per seed, and the
//! binary-under-test is separate from the cargo-test harness binary;
//! shelling out keeps the layering honest and matches how the
//! harness is used interactively.
//!
//! Running with output visible:
//!     cargo test --test external_validation -- --nocapture
//!
//! Regeneration (requires R+renv for any r-pomp cases):
//!     CAMDL_REGEN_EXTERNAL=1 cargo test --test external_validation -- --nocapture
//!
//! See docs/dev/testing.md §L9 and
//! docs/dev/proposals/2026-04-23-external-validation-harness.md.

use std::path::PathBuf;
use std::process::Command;

/// Walk up from the test binary's CWD to find the workspace root
/// (identified by `Cargo.toml` + `tests/external/cases/`). `cargo test`
/// sets CWD to the crate root already, so this is usually a no-op
/// lookup of `./tests/external/cases/`.
fn workspace_root() -> PathBuf {
    let cwd = std::env::current_dir().expect("cwd");
    let mut cur = cwd.as_path();
    loop {
        if cur.join("tests/external/cases").is_dir() {
            return cur.to_path_buf();
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => panic!(
                "could not locate tests/external/cases/ starting from {}",
                cwd.display()
            ),
        }
    }
}

fn harness_bin() -> PathBuf {
    // CARGO_BIN_EXE_external-harness is set by cargo when the
    // external-harness binary is a dev-dependency of this crate.
    // If that's not available (e.g. running the test file outside of
    // cargo's harness), fall back to a target/debug probe.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_external-harness") {
        return PathBuf::from(p);
    }
    // Fallback: look in target/{debug,release}/external-harness.
    let root = workspace_root();
    for profile in ["debug", "release"] {
        let p = root.join("rust").join("target").join(profile).join("external-harness");
        if p.exists() { return p; }
        let p = root.join("target").join(profile).join("external-harness");
        if p.exists() { return p; }
    }
    panic!(
        "external-harness binary not found. Build it first with: \
         cargo build -p external-harness"
    );
}

/// Locate the workspace-built `camdl` binary so the harness can spawn it
/// without depending on `camdl` being on `PATH`. The case manifests name
/// the binary as the bare `camdl`; under `cargo test`/CI nothing installs
/// it to a PATH directory (only `make install` does), so without this the
/// harness fails with `spawn camdl: No such file or directory`.
///
/// The harness binary (`external-harness`) and `camdl` are siblings in the
/// same `target/<profile>/` directory under `cargo test --workspace` (what
/// `make test-rust` runs). Prefer the sibling of the resolved harness
/// binary; fall back to a `target/{release,debug}/` probe. The resolved
/// path is handed to the harness via the `CAMDL` env var, matching the
/// `CAMDL=${CAMDL:-camdl}` convention in `tests/test_ocaml_to_rust.sh`.
fn camdl_bin() -> PathBuf {
    // Explicit override. `make test-rust` points `CAMDL` at the freshly-built
    // release binary (`build-rust` always rebuilds it) so the heavy pfilter
    // cases — 2000 particles × 10 replicates over 1096 weeks, twice — run in
    // release, not the ~10-30× slower debug build cargo would otherwise pick as
    // the harness sibling. Interactive `cargo test` with no override falls
    // through to the always-fresh debug sibling below.
    if let Ok(p) = std::env::var("CAMDL") {
        if !p.is_empty() {
            let pb = PathBuf::from(&p);
            if pb.exists() {
                return pb;
            }
        }
    }
    // Sibling of the harness binary, same profile directory.
    let harness = harness_bin();
    if let Some(dir) = harness.parent() {
        let sibling = dir.join("camdl");
        if sibling.exists() {
            return sibling;
        }
    }
    // Fallback probe. Prefer release (make test-rust always builds it via
    // `build-rust`) over debug.
    let root = workspace_root();
    for profile in ["release", "debug"] {
        let p = root.join("rust").join("target").join(profile).join("camdl");
        if p.exists() { return p; }
        let p = root.join("target").join(profile).join("camdl");
        if p.exists() { return p; }
    }
    panic!(
        "camdl binary not found next to the external-harness binary or under \
         target/{{release,debug}}/. Build it first with: cargo build -p cli \
         (or run via `make test-rust`, which builds the release binaries)."
    );
}

#[test]
fn run_all_cases() {
    let root = workspace_root();
    let bin = harness_bin();
    let camdl = camdl_bin();

    let cases_root = root.join("tests/external/cases");
    let status = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .args(["run-all", "--root"])
        .arg(&cases_root)
        .current_dir(&root)
        // The harness spawns the `camdl` token from each case.toml; point
        // it at the workspace-built binary so the test does not depend on
        // `camdl` being on PATH (it is not under `cargo test`/CI).
        .env("CAMDL", &camdl)
        .status()
        .expect("spawn external-harness");

    assert!(
        status.success(),
        "external-harness run-all failed (exit {:?}). Rerun with --nocapture to see per-case detail:\n    \
         cargo test --test external_validation -- --nocapture",
        status.code()
    );
}
