//! The output manifest: the JSON file a command that stores results writes at
//! `--manifest PATH`, listing every result its invocation produced or reused,
//! each result file under its logical name with its SHA-256 and size. A
//! workflow tool declares the manifest as a rule's output and reads run ids
//! and file names from it. The format is `docs/camdl-run-spec.md` §2.10.
//!
//! A command claims its path ([`claim`]) after its arguments have parsed and
//! before its first call to the store: the path is checked and any file there
//! is removed, so a command that then fails, or is killed, leaves no
//! manifest. Once the command has succeeded it writes the manifest
//! ([`ManifestPath::write`]) through a temporary file renamed over the path,
//! so a reader sees no manifest or a whole one.
//!
//! The manifest's bytes are a function of the invocation's ordered results,
//! their attributes and sources, their result files' names, digests and sizes, the `--label`
//! and the build. It holds no time, outcome or path, the store root and the
//! results' folders included, so a cache hit writes the bytes the computation
//! wrote and one command against two store roots writes one manifest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use store::Key;

/// The manifest's `schema` value.
pub(crate) const SCHEMA: &str = "camdl.output-manifest/v2";

/// One command's output manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OutputManifest {
    /// [`SCHEMA`].
    pub schema: String,
    /// The camdl that wrote the manifest.
    pub camdl: BuildRecord,
    /// The command's `--label`, recorded and never resolved.
    pub label: Option<String>,
    /// Each distinct result the invocation listed, in the order it first
    /// listed it.
    pub results: Vec<ManifestEntry>,
}

/// The build of the camdl that wrote a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildRecord {
    /// The version and source state, as `camdl --version` prints it.
    pub version: String,
    /// The 32-hex build fingerprint compiled into the binary.
    pub engine_id: String,
    /// The hash of the build and the environment switches that every run id
    /// this build computes holds.
    pub ambient_key: Key,
}

/// One result of a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestEntry {
    pub kind: String,
    pub run_id: Key,
    /// The result this one is a by-product of; `None` for any other result.
    pub upstream: Option<Key>,
    /// The kind's contract version (run spec §2.9).
    pub contract: u16,
    /// The attributes the invocation record gave the result.
    pub attributes: serde_json::Value,
    /// For a compiled model, the run id of the model source the invocation
    /// recorded for it, which is another entry of the manifest; absent for
    /// every other kind, and for a compiled model with no recorded source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Key>,
    /// Every result file of the result, by logical name. Provenance and
    /// attempt files are not listed.
    pub files: BTreeMap<String, ManifestFile>,
}

/// One result file of a manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifestFile {
    /// The stored name, relative to the result's folder.
    pub file: String,
    /// The SHA-256 of the file's bytes, as its `run.json` records it.
    pub sha256: Key,
    pub bytes: u64,
}

impl OutputManifest {
    /// The manifest of `invocation`, an ended invocation of `store`. Refused
    /// when any result it listed did not complete, and when a result holds a
    /// file its kind's contract does not name.
    pub(crate) fn of(store: &store::Store, invocation: &store::Invocation) -> Result<OutputManifest, String> {
        let entries = invocation.entries();
        let named = |completed: bool| -> Vec<String> {
            let mut v: Vec<String> =
                entries.iter().filter(|e| e.2 == completed).map(|(k, id, _)| format!("{k} {}", id.short12())).collect();
            v.dedup();
            v
        };
        let missing = named(false);
        if !missing.is_empty() {
            let stored = named(true);
            return Err(format!(
                "no manifest or links are written, because {} result(s) the command asked for were not stored: {}. {}; \
                 ask for the missing results again, and a later request computes them",
                missing.len(),
                missing.join(", "),
                if stored.is_empty() {
                    "Nothing else was stored".to_string()
                } else {
                    format!("The rest are stored: {}", stored.join(", "))
                },
            ));
        }
        let outputs = invocation.outputs(store).map_err(|e| e.to_string())?;
        Self::of_outputs(store, invocation.label(), outputs)
    }

    /// The manifest of the completed results `run_ids` of `store`, in that
    /// order with repeats dropped, each with the attributes the invocation
    /// that answers for it gives it ([`store::Store::listings`]): for a
    /// command, such as `compare`, that reads results its own invocation
    /// record does not list.
    pub(crate) fn of_results(store: &store::Store, run_ids: &[Key]) -> Result<OutputManifest, String> {
        let mut seen = std::collections::BTreeSet::new();
        let mut outputs = Vec::new();
        for id in run_ids.iter().filter(|id| seen.insert(**id)) {
            let located = store.locate(*id).map_err(|e| e.to_string())?;
            let files = located
                .result_files()
                .into_iter()
                .map(|(name, (sha256, bytes))| (name.to_string(), store::OutputFile { sha256, bytes }))
                .collect();
            let listings = store.listings(&store::Handle::IdPrefix(id.to_hex())).map_err(|e| e.to_string())?;
            let answering = listings.answering().map(|l| l.invocation());
            let attributes =
                answering.and_then(|i| i.attributes_of(located.kind(), *id)).unwrap_or(serde_json::Value::Null);
            let source = answering.and_then(|i| i.source_of(located.kind(), *id));
            outputs.push(store::Output {
                kind: located.kind().to_string(),
                run_id: *id,
                upstream: located.upstream(),
                attributes,
                source,
                files,
            });
        }
        Self::of_outputs(store, None, outputs)
    }

    fn of_outputs(store: &store::Store, label: Option<&str>, outputs: Vec<store::Output>) -> Result<OutputManifest, String> {
        let mut results = Vec::with_capacity(outputs.len());
        for o in outputs {
            let contract = crate::result_files::contract(&o.kind)
                .ok_or_else(|| format!("internal error: the result {} is of kind `{}`, which has no file contract", o.run_id.short12(), o.kind))?;
            let mut files = BTreeMap::new();
            for (stored, f) in o.files {
                let logical = contract.logical(&stored).ok_or_else(|| {
                    format!(
                        "internal error: the {} result {} holds `{stored}`, which its kind's file contract does not name",
                        o.kind,
                        o.run_id.short12()
                    )
                })?;
                let file = ManifestFile { file: stored.clone(), sha256: f.sha256, bytes: f.bytes };
                if let Some(other) = files.insert(logical.clone(), file) {
                    return Err(format!(
                        "internal error: the {} result {} holds `{}` and `{stored}`, which share the logical name `{logical}`",
                        o.kind,
                        o.run_id.short12(),
                        other.file
                    ));
                }
            }
            let attributes = manifest_attributes(&o.kind, o.attributes);
            results.push(ManifestEntry {
                kind: o.kind,
                run_id: o.run_id,
                upstream: o.upstream,
                contract: contract.version,
                attributes,
                source: o.source.map(|s| s.run_id),
                files,
            });
        }
        Ok(OutputManifest {
            schema: SCHEMA.into(),
            camdl: BuildRecord {
                version: crate::version::VERSION_SHORT.into(),
                engine_id: crate::version::ENGINE_ID.into(),
                ambient_key: store.ambient_key(),
            },
            label: label.map(str::to_string),
            results,
        })
    }

    /// Reads the manifest at `path`, given as the argument of `flag`, which
    /// each refusal names. A file that is not JSON, a JSON file whose
    /// `schema` is not an output manifest's, and a manifest of a version
    /// other than [`SCHEMA`]'s are refused, each saying what was found.
    #[expect(clippy::disallowed_methods, reason = "reads a manifest a command wrote, which names stored results by run id; the results are read through the store")]
    pub(crate) fn read(path: &Path, flag: &str) -> Result<OutputManifest, String> {
        let shown = path.display();
        let bytes = std::fs::read(path).map_err(|e| format!("{flag} {shown}: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{flag} {shown}: not an output manifest: {e}"))?;
        let schema = value.get("schema").and_then(|s| s.as_str()).unwrap_or("");
        if !schema.starts_with("camdl.output-manifest/") {
            return Err(format!("{flag} {shown}: not an output manifest (its `schema` is {})", value.get("schema").map_or("absent".to_string(), |s| s.to_string())));
        }
        if schema != SCHEMA {
            return Err(format!("{flag} {shown}: an output manifest of version `{schema}`; this camdl reads `{SCHEMA}`"));
        }
        serde_json::from_value(value).map_err(|e| format!("{flag} {shown}: a malformed output manifest: {e}"))
    }

    /// The manifest as written: pretty JSON with a final newline.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = serde_json::to_vec_pretty(self).expect("a manifest serializes");
        bytes.push(b'\n');
        bytes
    }
}

/// A result's attributes as the manifest records them: the invocation
/// record's, less any that hold a path as it was typed. A data table's
/// `path` is the `--data` argument, which names one file differently from
/// another working directory, so the manifest keeps only its `stream`. Every
/// other kind's attributes hold names and values.
pub(crate) fn manifest_attributes(kind: &str, mut attributes: serde_json::Value) -> serde_json::Value {
    let link = match kind {
        k if k == <crate::camdl_store::DataTable as store::Artifact>::KIND => Some("path"),
        _ => None,
    };
    if let (Some(field), Some(map)) = (link, attributes.as_object_mut()) {
        map.remove(field);
    }
    attributes
}

/// A `--manifest` path a command has claimed: checked, and cleared of any
/// earlier manifest.
#[derive(Debug, Clone)]
pub(crate) struct ManifestPath(PathBuf);

impl ManifestPath {
    /// Claims `path`: refuses a directory, a path whose parent directory
    /// does not exist, and a file that is not an output manifest, and
    /// removes an earlier manifest at the path. An empty file counts as a
    /// manifest's place, and anything else is left as it is: a mistyped
    /// `--manifest data.tsv` must never delete the data.
    pub(crate) fn claim(path: &Path) -> Result<ManifestPath, String> {
        if path.is_dir() {
            return Err(format!("--manifest {}: is a directory", path.display()));
        }
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        if !parent.is_dir() {
            return Err(format!("--manifest {}: the directory {} does not exist", path.display(), parent.display()));
        }
        if !holds_a_manifest_or_nothing(path)? {
            return Err(format!("--manifest {}: {} holds a file that is not an output manifest", path.display(), path.display()));
        }
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("--manifest {}: cannot remove the earlier manifest: {e}", path.display())),
        }
        Ok(ManifestPath(path.to_path_buf()))
    }

    /// Writes `manifest` at the claimed path, atomically.
    pub(crate) fn write(&self, manifest: &OutputManifest) -> Result<(), String> {
        self.write_bytes(&manifest.to_bytes())
    }

    /// The claimed path.
    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    /// Writes the manifest of the results `run_ids` of `store` at the
    /// claimed path, atomically ([`OutputManifest::of_results`]).
    pub(crate) fn write_results(&self, store: &store::Store, run_ids: &[Key]) -> Result<(), String> {
        let manifest = OutputManifest::of_results(store, run_ids).map_err(|e| format!("--manifest {}: {e}", self.0.display()))?;
        self.write_bytes(&manifest.to_bytes())
    }

    /// Writes `bytes` at the claimed path through a temporary file in its
    /// directory, renamed over it.
    fn write_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        store::write_file_atomic(&self.0, bytes).map_err(|e| format!("--manifest {}: {e}", self.0.display()))
    }
}

/// Whether `path` holds nothing, an empty file, or a JSON object whose
/// `schema` names an output manifest of any version.
#[expect(clippy::disallowed_methods, reason = "reads the file at the `--manifest` path, an export the command is about to replace, to refuse replacing one that is not a manifest")]
fn holds_a_manifest_or_nothing(path: &Path) -> Result<bool, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(e) => return Err(format!("--manifest {}: cannot read the file there: {e}", path.display())),
    };
    if bytes.is_empty() {
        return Ok(true);
    }
    let schema = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.get("schema").and_then(|s| s.as_str()).map(str::to_string));
    Ok(schema.is_some_and(|s| s.starts_with("camdl.output-manifest/")))
}

/// Claims `path` when one is given, or fails the command with the refusal.
pub(crate) fn claim(out: &crate::out::Out, path: Option<&Path>) -> Option<ManifestPath> {
    path.map(|p| ManifestPath::claim(p).unwrap_or_else(|e| out.fail(format!("error: {e}"))))
}

#[cfg(test)]
mod tests;
