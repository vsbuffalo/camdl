//! Byte-digest and path-label helpers shared across the CLI.
//!
//! # Not run identity (gh#241)
//!
//! Artifact run identity is built exclusively by the `runid`/`resolve` paths
//! (`resolve::resolve_trajectory`, `resolve::model_identity_hex`,
//! `cas::fit_level_hash`, the `runid` factored levels). NOTHING in this module
//! participates in a `run_id`, and there is no longer any "semantic digest"
//! helper here: the legacy `model_hash` / `fit_content_hash` were retired in
//! favor of the `runid` model / fit-level digests. The functions here are one
//! of two kinds:
//!
//! - **byte digests**: [`sha256_hex`], [`file_hash`] — hash some bytes.
//! - **path labels**: [`path_stem_slug`], [`slug`] — make a directory name
//!   human-readable; the content hash that follows is the authoritative key.
//!
//! **DO NOT** reintroduce a run-identity or "model / fit digest" construction
//! here. A field that changes stored bytes is identity and must re-key through
//! `runid`; a re-encoding of the same values is presentation. The guard test
//! below fails if `model_hash(` / `fit_content_hash(` reappears as a call in
//! the crate's production source.

use sha2::{Sha256, Digest};

/// Full 64-char SHA-256 of a byte slice, hex-encoded. Used where the
/// caller wants a full content hash (e.g. fit_toml_hash in the top-level
/// fit run record); the 8-char truncated form is only appropriate when
/// the hash is paired with a richer identifier.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Hash the contents of a file (first 4 bytes of SHA256, 8 hex chars).
/// Returns `None` when the file can't be read — callers use this to
/// surface `<unreadable>` in provenance records rather than failing
/// the whole run.
///
/// Shared between simulate (data-file hashing for run metadata) and fit
/// (data-file hashing for per-stage provenance). Was
/// `fit::provenance::file_content_hash` before the 2026-04-19 unification.
pub fn file_hash(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(hex::encode(&Sha256::digest(&bytes)[..4]))
}

/// Extract a directory-safe stem from a file path: basename without
/// extension(s), slugified. Used to label fit / sim output
/// directories so `ls output/fits/` shows recognisable names alongside
/// their content hashes. Multi-dot extensions (`foo.ir.json` →
/// `foo`) collapse by stripping from the first dot.
pub fn path_stem_slug(path: &str) -> Option<String> {
    let name = std::path::Path::new(path).file_name()?.to_str()?;
    let stem = name.split('.').next().filter(|s| !s.is_empty())?;
    Some(slug(stem))
}

/// Convert a scenario name to a filesystem-safe slug: lowercase, non-[a-z0-9_] → '_'.
pub fn slug(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // There is no `model_hash` / `fit_content_hash` / `sim_hash` /
    // `scen_hash` / `canonical_params` here anymore. A run's identity is the
    // factored `runid` identity (`runid::run_id` over the per-level hashes; see
    // `crate::resolve::resolve_trajectory`), the model identity is
    // `crate::resolve::model_identity_hex`, and the fit-level identity is
    // `crate::fit::cas::fit_level_hash`.

    // ── legacy-identity guard (gh#241) ────────────────────────────────────────

    /// gh#241 — the legacy semantic digests are retired. Scan every `.rs` under
    /// the cli crate's `src/` (except this module and `args/`) and assert NO
    /// production call to `model_hash(` / `fit_content_hash(` reappears. Deleting
    /// the functions means the compiler already rejects a stray *call*; this
    /// guard additionally catches a re-`fn`-definition or a method of the same
    /// name sneaking back in as a run-identity consumer. Test code
    /// (`#[cfg(test)]` / `#[test]`) and comment lines are excluded.
    #[test]
    fn legacy_identity_helpers_are_gone() {
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let banned = ["model_hash(", "fit_content_hash("];

        let mut offenders: Vec<String> = Vec::new();
        let mut stack = vec![src_root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let rel = path.strip_prefix(&src_root).unwrap()
                    .to_string_lossy().replace('\\', "/");
                // Skip this module (the guard's own banned strings) and `args/`
                // (clap structs never hash).
                if rel == "hashing.rs" || rel.starts_with("args/") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut in_test = false;
                for line in text.lines() {
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("#[cfg(test)]")
                        || trimmed.starts_with("#[test]")
                        || trimmed.starts_with("mod tests")
                    {
                        in_test = true;
                    }
                    if in_test || trimmed.starts_with("//") {
                        continue;
                    }
                    for b in &banned {
                        if line.contains(b) {
                            offenders.push(format!("{}: {}", rel, line.trim()));
                        }
                    }
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "legacy semantic digests are retired (gh#241) — these production \
             call sites must use `runid`/`resolve` identity instead:\n{}",
            offenders.join("\n")
        );
    }

    // ── slug ─────────────────────────────────────────────────────────────────

    #[test]
    fn slug_alphanumeric_passthrough() {
        assert_eq!(slug("baseline"), "baseline");
        assert_eq!(slug("with_sia"), "with_sia");
    }

    #[test]
    fn slug_lowercases() {
        assert_eq!(slug("WithSIA"), "withsia");
    }

    #[test]
    fn slug_replaces_spaces_and_specials() {
        assert_eq!(slug("with sia!"), "with_sia_");
        assert_eq!(slug("r0=3.0"), "r0_3_0");
    }

    // ── file_hash ──────────────────────────────────────────────────────────

    #[test]
    fn file_hash_returns_8_hex() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_hash_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::write(&tmp, b"hello world").unwrap();
        let h = file_hash(tmp.to_str().unwrap()).unwrap();
        assert_eq!(h.len(), 8, "file_hash should return 8 hex chars");
        // SHA256("hello world")[..4] is b94d27b9 in hex.
        assert_eq!(h, "b94d27b9");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn file_hash_returns_none_for_missing() {
        assert!(file_hash("/does/not/exist/at/all").is_none());
    }
}
