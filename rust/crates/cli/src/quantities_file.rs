//! A **quantities vocabulary file**: an ordinary `.camdl` file containing only
//! a `quantities { }` block, supplied at the point of use (`simulate
//! --quantities`, `fit predict --quantities`) and compiled against the model it
//! is applied to. It REPLACES the model's own block wholesale — a merge rule
//! would be a silent-precedence surface.
//!
//! Proposal: `docs/dev/proposals/2026-08-19-quantities-as-a-separable-layer.md`.
//!
//! ## Why this type exists rather than a bare `PathBuf`
//!
//! Two vocabularies applied to one fit produce two different reporting tables.
//! Those tables are written beside the run (`quantities/<name>.tsv` +
//! `quantities.json`), at a path that until now was fixed — so the second run
//! would overwrite the first at one address, and a reader could not tell which
//! vocabulary produced the table it is holding. That is the collision class
//! fixed twice already (gh#626 `--to`, gh#641 `--init-state`): an
//! output-determining input that does not reach the key.
//!
//! So the vocabulary's **content digest keys the artifact**: the tables land in
//! `quantities-<key8>/` with a `quantities-<key8>.json` manifest. The digest is
//! over the file's BYTES, never its path — an in-place edit re-keys (the point
//! of the feature is that a corrected formula produces a new table), and two
//! copies of one vocabulary share an address.
//!
//! Model/fit identity is deliberately untouched: `quantities` are excluded from
//! `Model::hash_into` (pinned by `runid`'s `ir_quantities_excluded_from_hash`),
//! so the trajectory and the posterior a vocabulary is read off do not move. It
//! is the *report* that is keyed, not the run.

use std::path::{Path, PathBuf};

/// A loaded vocabulary file: where it came from, what it says, and the digest
/// that keys anything computed from it.
#[derive(Debug, Clone)]
pub(crate) struct QuantitiesOverride {
    /// The path as written on the command line. Provenance only — recorded in
    /// the manifest so a table can be traced back, never part of the key.
    pub path: PathBuf,
    /// The file's bytes, handed to `camdlc --quantities`.
    pub bytes: Vec<u8>,
    /// `sha256(bytes)`, lowercase hex. The key.
    pub digest: String,
}

impl QuantitiesOverride {
    /// Read and digest a vocabulary file. Errors name the path — a mistyped
    /// `--quantities` must not degrade into "the model's own block was used",
    /// which is the same silent-fallback the whole feature exists to avoid.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| {
            format!("cannot read the quantities file {}: {e}", path.display())
        })?;
        let digest = crate::hashing::sha256_hex(&bytes);
        Ok(Self { path: path.to_path_buf(), bytes, digest })
    }

    /// The 8-hex artifact key — the same width every other camdl path segment
    /// uses (`scen_h8`, `param_h8`).
    pub fn key8(&self) -> &str {
        &self.digest[..8]
    }

    /// Provenance for the manifest: which vocabulary produced this table, by
    /// path AND by full content digest. The digest is what makes the record
    /// checkable after the file moves or changes.
    pub fn provenance(&self) -> serde_json::Value {
        serde_json::json!({
            "file": self.path.display().to_string(),
            "sha256": self.digest,
        })
    }
}

/// The output subdirectory for a run's quantity TSVs: `quantities` for the
/// model's own block, `quantities-<key8>` for a supplied vocabulary.
///
/// One function, shared by `simulate --quantities-out` and `fit predict`, so
/// the two cannot key their artifacts differently — the failure mode of a
/// second copy is that one command collides while the other does not, which is
/// invisible until two tables disagree.
pub(crate) fn quantities_dir_name(q: Option<&QuantitiesOverride>) -> String {
    match q {
        None => "quantities".to_string(),
        Some(q) => format!("quantities-{}", q.key8()),
    }
}

/// The manifest filename that pairs with [`quantities_dir_name`].
pub(crate) fn quantities_manifest_name(q: Option<&QuantitiesOverride>) -> String {
    format!("{}.json", quantities_dir_name(q))
}

/// Record which vocabulary produced a manifest, in the manifest itself.
///
/// The 8-hex key in the directory name says two tables are different; the
/// `vocabulary` object says *which file* each one came from and pins its full
/// digest, so a table found later can be traced to the formulas that made it
/// without guessing. Absent (no key at all) when the model's own block was
/// used — the historical bytes are unchanged for every existing run.
pub(crate) fn stamp_provenance(
    manifest_json: &str,
    q: Option<&QuantitiesOverride>,
) -> Result<String, String> {
    let Some(q) = q else { return Ok(manifest_json.to_string()) };
    let mut m: serde_json::Value = serde_json::from_str(manifest_json)
        .map_err(|e| format!("parsing the quantities manifest: {e}"))?;
    m["vocabulary"] = q.provenance();
    serde_json::to_string_pretty(&m)
        .map_err(|e| format!("serializing the quantities manifest: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("camdl_qfile_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The key is the file's BYTES, not its path: two copies of one vocabulary
    /// share an address, and an in-place edit moves it.
    #[test]
    fn key_is_content_not_path() {
        let d = tmpdir("content");
        let a = write(&d, "a.camdl", "quantities { x = final(S) }\n");
        let b = write(&d, "b.camdl", "quantities { x = final(S) }\n");
        let qa = QuantitiesOverride::load(&a).unwrap();
        let qb = QuantitiesOverride::load(&b).unwrap();
        assert_eq!(qa.digest, qb.digest, "two copies of one vocabulary are one address");

        // In-place edit: same path, different bytes → a different key.
        std::fs::write(&a, "quantities { x = final(R) }\n").unwrap();
        let qa2 = QuantitiesOverride::load(&a).unwrap();
        assert_ne!(qa.digest, qa2.digest, "an in-place edit must re-key");
        assert_ne!(qa.key8(), qa2.key8(), "…including the 8-hex path segment");
    }

    /// The directory and its manifest agree, and the no-override names are the
    /// historical ones (a model's own block keeps writing `quantities/`).
    #[test]
    fn artifact_names_pair_and_default_to_the_historical_ones() {
        let d = tmpdir("names");
        let p = write(&d, "v.camdl", "quantities { x = final(S) }\n");
        let q = QuantitiesOverride::load(&p).unwrap();
        assert_eq!(quantities_dir_name(None), "quantities");
        assert_eq!(quantities_manifest_name(None), "quantities.json");
        assert_eq!(quantities_dir_name(Some(&q)), format!("quantities-{}", q.key8()));
        assert_eq!(
            quantities_manifest_name(Some(&q)),
            format!("quantities-{}.json", q.key8())
        );
    }

    /// A missing vocabulary is an error naming the path, never a quiet fall
    /// back to the model's own block.
    #[test]
    fn a_missing_vocabulary_errors_naming_the_path() {
        let err = QuantitiesOverride::load(Path::new("/nonexistent/vocab.camdl")).unwrap_err();
        assert!(err.contains("vocab.camdl"), "error must name the file: {err}");
    }
}
