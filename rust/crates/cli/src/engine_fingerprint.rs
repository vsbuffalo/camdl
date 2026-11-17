//! The engine's build fingerprint, `ENGINE_ID`: what a run's identity records
//! as "the engine that computed it".
//!
//! This file is compiled twice. `build.rs` includes it (`#[path]`) and calls
//! [`gather`] and [`fingerprint`] to embed `CAMDL_ENGINE_ID` and the readable
//! version; `cli`'s unit tests compile it as a module to exercise the same
//! functions on injected inputs. It therefore uses `std` and `sha2` only (the
//! build script's one build-dependency), and nothing from the `cli` crate.
//!
//! `ENGINE_ID` is a 128-bit digest of the inputs a build depends on: the
//! source fingerprint (the commit plus a digest of the uncommitted changes to
//! the files that build the binary, or, without git, a digest of those files),
//! `rust/Cargo.lock`, the resolved Cargo features of the crates in the binary,
//! the build profile, `rustc -vV`, the encoded `RUSTFLAGS` and the target
//! triple. When any of them cannot be determined the fingerprint says so
//! ([`Fingerprint::missing`]); such a binary computes and records results but
//! never reads a stored one (`cli::version::ENGINE_REUSE`), and its id carries
//! a per-build nonce so it never coincides with another build's.
//!
//! It is a build fingerprint under one machine image, not a complete identity
//! of the binary: the system toolchain used for native code and the
//! dynamically linked libraries (the operating system's `exp`, `ln`, `pow`)
//! are not in it. Proposal:
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` §1a.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

/// The state of the source tree the binary is built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceState {
    /// A git checkout: the commit, and a digest of the uncommitted changes to
    /// the build inputs (`None` when there are none).
    Git { commit: String, short: String, dirty: Option<String> },
    /// No git: a digest of every build input's path and bytes.
    NoGit { digest: String },
}

/// Everything the fingerprint is computed from. `None` is a component that
/// could not be determined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInputs {
    /// `CARGO_PKG_VERSION`, the release version.
    pub pkg_version: String,
    pub source: Option<SourceState>,
    /// SHA-256 hex of `rust/Cargo.lock`.
    pub cargo_lock: Option<String>,
    /// The resolved Cargo features of every workspace crate in the binary that
    /// has any, as `crate:feature`, sorted.
    pub features: Option<Vec<String>>,
    /// The build profile: its directory name, `PROFILE`, `OPT_LEVEL`,
    /// `DEBUG` and whether debug assertions are on.
    pub profile: Option<String>,
    /// `rustc -vV`, verbatim.
    pub rustc: Option<String>,
    /// `CARGO_ENCODED_RUSTFLAGS`, verbatim (0x1f-separated).
    pub rustflags: Option<String>,
    pub target: Option<String>,
}

/// The computed fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint {
    /// 32 lowercase hex characters (128 bits).
    pub engine_id: String,
    /// The readable suffix after `<version>+`: `abc1234`,
    /// `abc1234.dirty-3f9c21e0`, `nogit-1a2b3c4d` or `unknown`.
    pub source_label: String,
    /// The components that could not be determined; empty for a reusable
    /// identity.
    pub missing: Vec<&'static str>,
}

/// Compute the fingerprint. `nonce` is called only when a component is
/// missing, and its value is folded in so that no two such builds share an id.
pub fn fingerprint(inputs: &BuildInputs, nonce: impl FnOnce() -> String) -> Fingerprint {
    let mut h = Sha256::new();
    let mut put = |name: &str, value: Option<&str>| {
        h.update(name.as_bytes());
        h.update([0u8]);
        match value {
            Some(v) => {
                h.update([1u8]);
                h.update((v.len() as u64).to_le_bytes());
                h.update(v.as_bytes());
            }
            None => h.update([0u8]),
        }
    };
    put("camdl-engine-id", Some("1"));
    put("pkg_version", Some(&inputs.pkg_version));
    let source = inputs.source.as_ref().map(|s| match s {
        SourceState::Git { commit, dirty: None, .. } => format!("git {commit}"),
        SourceState::Git { commit, dirty: Some(d), .. } => format!("git {commit} dirty {d}"),
        SourceState::NoGit { digest } => format!("nogit {digest}"),
    });
    put("source", source.as_deref());
    put("cargo_lock", inputs.cargo_lock.as_deref());
    let features = inputs.features.as_ref().map(|f| f.join(","));
    put("features", features.as_deref());
    put("profile", inputs.profile.as_deref());
    put("rustc", inputs.rustc.as_deref());
    put("rustflags", inputs.rustflags.as_deref());
    put("target", inputs.target.as_deref());

    let mut missing = Vec::new();
    for (name, absent) in [
        ("source", inputs.source.is_none()),
        ("cargo_lock", inputs.cargo_lock.is_none()),
        ("features", inputs.features.is_none()),
        ("profile", inputs.profile.is_none()),
        ("rustc", inputs.rustc.is_none()),
        ("rustflags", inputs.rustflags.is_none()),
        ("target", inputs.target.is_none()),
    ] {
        if absent {
            missing.push(name);
        }
    }
    if !missing.is_empty() {
        put("unreusable_nonce", Some(&nonce()));
    }
    let digest = h.finalize();
    let engine_id = hex(&digest[..16]);
    let source_label = match &inputs.source {
        Some(SourceState::Git { short, dirty: None, .. }) => short.clone(),
        Some(SourceState::Git { short, dirty: Some(d), .. }) => {
            format!("{short}.dirty-{}", &d[..8.min(d.len())])
        }
        Some(SourceState::NoGit { digest }) => format!("nogit-{}", &digest[..8.min(digest.len())]),
        None => "unknown".to_string(),
    };
    Fingerprint { engine_id, source_label, missing }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Whether a repository-relative path (`/`-separated) is one of the files the
/// binary is built from: `ir/VERSION`, and under `rust/` the workspace and
/// crate `Cargo.toml` and `build.rs` files and every file in a `src/`
/// directory, less the test-only files kept there (any file named `tests.rs`
/// or under a `tests` directory, each declared `#[cfg(test)]`). Tests and
/// benches outside `src/` are not build inputs, so editing one leaves
/// `ENGINE_ID` unchanged. `Cargo.lock` is its own component.
pub fn is_build_input(rel: &str) -> bool {
    if rel == "ir/VERSION" {
        return true;
    }
    let Some(rest) = rel.strip_prefix("rust/") else { return false };
    let parts: Vec<&str> = rest.split('/').collect();
    // Strip the crate prefix: `crates/<name>/…` or the workspace root itself.
    let in_crate: &[&str] = match parts.as_slice() {
        ["crates", name, tail @ ..] if !name.is_empty() && !tail.is_empty() => tail,
        ["crates", ..] => return false,
        other => other,
    };
    match in_crate {
        ["Cargo.toml"] | ["build.rs"] => true,
        ["src", tail @ ..] if !tail.is_empty() => {
            let (file, dirs) = tail.split_last().expect("non-empty");
            *file != "tests.rs" && !dirs.contains(&"tests")
        }
        _ => false,
    }
}

/// The directories and files whose change must re-run the build script so
/// the fingerprint stays fresh: every crate's `src/`, `Cargo.toml` and
/// `build.rs`, the workspace's, `rust/Cargo.lock` and `ir/VERSION`. Only
/// paths that exist are returned: cargo re-runs a script on every build when
/// a watched path is absent.
pub fn watched_source_paths(repo_root: &Path) -> Vec<PathBuf> {
    let rust = repo_root.join("rust");
    let mut out = Vec::new();
    let mut crate_dirs = vec![rust.clone()];
    if let Ok(entries) = std::fs::read_dir(rust.join("crates")) {
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        crate_dirs.extend(dirs);
    }
    for dir in crate_dirs {
        for name in ["src", "Cargo.toml", "build.rs"] {
            out.push(dir.join(name));
        }
    }
    out.push(rust.join("Cargo.lock"));
    out.push(repo_root.join("ir").join("VERSION"));
    out.retain(|p| p.exists());
    out
}

/// Every build input under `repo_root`, as sorted repository-relative paths.
fn walk_build_inputs(repo_root: &Path) -> std::io::Result<Vec<String>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if let Ok(rel) = path.strip_prefix(root) {
                let rel = rel.to_string_lossy().replace('\\', "/");
                if is_build_input(&rel) {
                    out.push(rel);
                }
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    for p in watched_source_paths(repo_root) {
        if p.is_dir() {
            walk(repo_root, &p, &mut out)?;
        } else if let Ok(rel) = p.strip_prefix(repo_root) {
            let rel = rel.to_string_lossy().replace('\\', "/");
            if is_build_input(&rel) {
                out.push(rel);
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// A digest of the named files' paths and current bytes; a file that no
/// longer exists (an uncommitted deletion) enters as deleted.
fn digest_files(repo_root: &Path, rels: &[String]) -> std::io::Result<String> {
    let mut h = Sha256::new();
    for rel in rels {
        h.update((rel.len() as u64).to_le_bytes());
        h.update(rel.as_bytes());
        match std::fs::read(repo_root.join(rel)) {
            Ok(bytes) => {
                h.update([1u8]);
                h.update((bytes.len() as u64).to_le_bytes());
                h.update(&bytes);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => h.update([0u8]),
            Err(e) => return Err(e),
        }
    }
    Ok(hex(&h.finalize()))
}

fn git(repo_root: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git")
        .arg("--no-optional-locks")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

fn git_line(repo_root: &Path, args: &[&str]) -> Option<String> {
    let bytes = git(repo_root, args)?;
    let s = String::from_utf8(bytes).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Whether `repo_root` is the top of a git work tree (its own, not an
/// enclosing repository the source happens to sit in).
pub fn is_git_checkout(repo_root: &Path) -> bool {
    let Some(top) = git_line(repo_root, &["rev-parse", "--show-toplevel"]) else { return false };
    match (std::fs::canonicalize(top), std::fs::canonicalize(repo_root)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The source fingerprint of the tree at `repo_root`, or `None` when it
/// cannot be determined (a git checkout whose status cannot be read, or a
/// build input that cannot be read).
///
/// In a git checkout: the `HEAD` commit, and a digest of the build inputs that
/// differ from it (tracked edits, deletions, and untracked files git does not
/// ignore). Without git: a digest of every build input.
pub fn source_state(repo_root: &Path) -> Option<SourceState> {
    if !is_git_checkout(repo_root) {
        let files = walk_build_inputs(repo_root).ok()?;
        if files.is_empty() {
            return None;
        }
        return Some(SourceState::NoGit { digest: digest_files(repo_root, &files).ok()? });
    }
    let commit = git_line(repo_root, &["rev-parse", "HEAD"])?;
    let short = git_line(repo_root, &["rev-parse", "--short", "HEAD"])?;
    let status = git(
        repo_root,
        &[
            "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
            "--ignore-submodules=all", "--", "rust", "ir/VERSION",
        ],
    )?;
    let mut changed: Vec<String> = status
        .split(|b| *b == 0)
        .filter(|e| e.len() > 3)
        .filter_map(|e| std::str::from_utf8(&e[3..]).ok().map(str::to_string))
        .filter(|rel| is_build_input(rel))
        .collect();
    changed.sort();
    changed.dedup();
    let dirty = if changed.is_empty() { None } else { Some(digest_files(repo_root, &changed).ok()?) };
    Some(SourceState::Git { commit, short, dirty })
}

/// The git files whose change means `HEAD` moved: the work tree's `HEAD`,
/// the branch ref it names (or, when that ref is packed, its nearest existing
/// directory) and `packed-refs`. Existing paths only.
pub fn watched_git_paths(repo_root: &Path) -> Vec<PathBuf> {
    let abs = |p: String| {
        let p = PathBuf::from(p);
        if p.is_absolute() { p } else { repo_root.join(p) }
    };
    let (Some(git_dir), Some(common)) = (
        git_line(repo_root, &["rev-parse", "--git-dir"]).map(abs),
        git_line(repo_root, &["rev-parse", "--git-common-dir"]).map(abs),
    ) else {
        return Vec::new();
    };
    let head = git_dir.join("HEAD");
    let mut out = vec![head.clone(), common.join("packed-refs")];
    if let Ok(text) = std::fs::read_to_string(&head) {
        if let Some(r) = text.trim().strip_prefix("ref: ") {
            let mut p = common.join(r);
            while !p.exists() {
                match p.parent() {
                    Some(parent) if parent.starts_with(&common) && parent != common => p = parent.to_path_buf(),
                    _ => break,
                }
            }
            out.push(p);
        }
    }
    out.retain(|p| p.exists());
    out
}

/// Gather [`BuildInputs`] from a build script's environment. `env` reads one
/// variable; `rustc_vv` runs `rustc -vV`; `cli_features` are `cli`'s own
/// enabled features, as `cli:<name>`.
pub fn gather(
    repo_root: &Path,
    env: &dyn Fn(&str) -> Option<String>,
    rustc_vv: &dyn Fn() -> Option<String>,
    cli_features: Vec<String>,
) -> BuildInputs {
    // `sim`'s resolved features reach this script only through `links`
    // metadata (`sim/build.rs`): a build script sees its own package's
    // features, and `sim`'s are not `cli`'s (`io` enables `sim`'s defaults
    // whatever `cli` asks for).
    let features = env("DEP_CAMDL_SIM_FEATURES").map(|sim| {
        let mut all = cli_features;
        all.extend(sim.split(',').filter(|s| !s.is_empty()).map(|s| format!("sim:{s}")));
        all.sort();
        all
    });
    let profile = (|| {
        let out_dir = PathBuf::from(env("OUT_DIR")?);
        // OUT_DIR = <target>/[<triple>/]<profile-dir>/build/<pkg>-<hash>/out
        let profile_dir = out_dir.ancestors().nth(3)?.file_name()?.to_string_lossy().into_owned();
        Some(format!(
            "dir={profile_dir} profile={} opt_level={} debug={} debug_assertions={}",
            env("PROFILE")?,
            env("OPT_LEVEL")?,
            env("DEBUG")?,
            env("CARGO_CFG_DEBUG_ASSERTIONS").is_some(),
        ))
    })();
    BuildInputs {
        pkg_version: env("CARGO_PKG_VERSION").unwrap_or_default(),
        source: source_state(repo_root),
        cargo_lock: std::fs::read(repo_root.join("rust").join("Cargo.lock")).ok().map(|b| sha256_hex(&b)),
        features,
        profile,
        rustc: rustc_vv(),
        rustflags: env("CARGO_ENCODED_RUSTFLAGS"),
        target: env("TARGET"),
    }
}

#[cfg(test)]
mod tests;
