//! Embeds the version strings and the engine's build fingerprint.
//!
//! - `CAMDL_GIT_HASH`: `git rev-parse --short HEAD`, for the camdlc version
//!   handshake (`util::check_camdlc_version_once`). Unchanged by uncommitted
//!   edits, deliberately: the handshake compares commits, not builds.
//! - `CAMDL_ENGINE_ID`, `CAMDL_SOURCE_LABEL`, `CAMDL_ENGINE_UNREUSABLE`: the
//!   build fingerprint (`src/engine_fingerprint.rs`), the readable suffix of
//!   the version, and the components that could not be determined (empty for
//!   a reusable identity).
//!
//! The script re-runs when any build input changes (every crate's `src/`,
//! `Cargo.toml` and `build.rs`, the lock file, `ir/VERSION`) and when `HEAD`
//! moves, so the fingerprint is never stale. Features, profile, flags, target
//! and toolchain changes re-run it by themselves.

#[path = "src/engine_fingerprint.rs"]
mod engine_fingerprint;

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    // rust/crates/cli → the repository root.
    let repo_root: PathBuf = manifest.ancestors().nth(3).map(Path::to_path_buf).unwrap_or_else(|| manifest.clone());

    let hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&manifest)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());

    let date = std::process::Command::new("date")
        .args(["+%Y-%m-%d"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into());

    let env = |k: &str| std::env::var(k).ok();
    let rustc_vv = || {
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
        std::process::Command::new(rustc)
            .arg("-vV")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .filter(|s| !s.trim().is_empty())
    };
    let mut cli_features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(|n| format!("cli:{}", n.to_lowercase())))
        .collect();
    cli_features.sort();
    let inputs = engine_fingerprint::gather(&repo_root, &env, &rustc_vv, cli_features);
    let fp = engine_fingerprint::fingerprint(&inputs, || {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{nanos}-{}", std::process::id())
    });
    if !fp.missing.is_empty() {
        println!(
            "cargo:warning=camdl engine identity could not determine: {}; this build \
             computes every run and never reads a stored one",
            fp.missing.join(", ")
        );
    }

    println!("cargo:rustc-env=CAMDL_GIT_HASH={hash}");
    println!("cargo:rustc-env=CAMDL_BUILD_DATE={}", date.trim());
    println!("cargo:rustc-env=CAMDL_ENGINE_ID={}", fp.engine_id);
    println!("cargo:rustc-env=CAMDL_SOURCE_LABEL={}", fp.source_label);
    println!("cargo:rustc-env=CAMDL_ENGINE_UNREUSABLE={}", fp.missing.join(","));

    println!("cargo:rerun-if-changed=build.rs");
    for p in engine_fingerprint::watched_source_paths(&repo_root) {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    if engine_fingerprint::is_git_checkout(&repo_root) {
        for p in engine_fingerprint::watched_git_paths(&repo_root) {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }
}
