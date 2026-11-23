//! Completed results, dependencies on their files, and results in any status.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::artifact::{Artifact, Id};
use crate::error::{Error, Result};
use crate::fsutil::write_file_atomic;
use crate::hash::{CanonicalHasher, ContentAddressed, Key};
use crate::identity::{DepInfo, Identity};
use crate::invocation::ExportPath;
use crate::owner::{Liveness, Owner};
use crate::record::{read_record, FileEntry, IdentityFile, Lineage, Role, RunFile, IDENTITY_FILE, RUN_FILE};
use crate::run::read_checked;
use crate::storage::Inner;

/// A completed result. The only type that opens result files or makes a
/// [`Dep`].
pub struct Stored<A: Artifact> {
    pub(crate) id: Id<A>,
    pub(crate) dir: PathBuf,
    pub(crate) record: <A::Identity as Identity>::Record,
    pub(crate) run: RunFile,
    pub(crate) build: String,
}

impl<A: Artifact> Stored<A> {
    /// Reads the completed result in `dir`, whose run id is `run_id`.
    pub(crate) fn open_at(store: &Arc<Inner>, run_id: Key, dir: &Path) -> Result<Self> {
        let ident: IdentityFile = read_record(&dir.join(IDENTITY_FILE)).required(&dir.join(IDENTITY_FILE))?;
        let run: RunFile = read_record(&dir.join(RUN_FILE)).required(&dir.join(RUN_FILE))?;
        if ident.kind != A::KIND {
            return Err(Error::WrongKind { handle: dir.display().to_string(), expected: A::KIND, found: ident.kind });
        }
        let record = serde_json::from_value(ident.identity.clone())
            .map_err(|e| Error::Parse { what: dir.join(IDENTITY_FILE).display().to_string(), why: e.to_string() })?;
        let build = store.ambient_key_of(&ident.ambient).map(|k| k.to_hex()).unwrap_or_default();
        Ok(Stored { id: Id::from_key(run_id), dir: dir.to_path_buf(), record, run, build })
    }

    pub fn id(&self) -> &Id<A> {
        &self.id
    }

    /// The record form of the identity.
    pub fn identity(&self) -> &<A::Identity as Identity>::Record {
        &self.record
    }

    /// The result's folder, for display. Readers ask the record, not the path.
    pub fn folder(&self) -> &Path {
        &self.dir
    }

    /// The result file names, in name order.
    pub fn result_files(&self) -> Vec<&str> {
        self.run.files.iter().filter(|(_, e)| e.role == Role::Result).map(|(n, _)| n.as_str()).collect()
    }

    fn result_digest(&self, file: &str) -> Result<Key> {
        match self.run.files.get(file) {
            Some(e) if e.role == Role::Result => Ok(e.sha256.expect("a result file has a digest")),
            _ => Err(Error::BadFileName { name: format!("{file} (not a result file of {})", self.id.key().short12()) }),
        }
    }

    fn make_dep(&self, files: DepFiles, digest: Key, whole: Option<Key>) -> Dep<A> {
        Dep { id: self.id, files, digest, build: self.build.clone(), whole }
    }

    /// A dependency on one result file, its digest taken from the manifest.
    pub fn dep(&self, file: &str) -> Result<Dep<A>> {
        let digest = self.result_digest(file)?;
        Ok(self.make_dep(DepFiles::One(file.to_string()), digest, None))
    }

    /// A dependency on every result file; never a provenance file.
    pub fn dep_all(&self) -> Dep<A> {
        let names = self.result_files();
        let digest = files_digest(&self.run.files, &names).expect("result files have digests");
        self.make_dep(DepFiles::All, digest, None)
    }

    /// A dependency on the named result files, as one.
    pub fn dep_files(&self, files: &[&str]) -> Result<Dep<A>> {
        let mut names: Vec<String> = files.iter().map(|s| s.to_string()).collect();
        names.sort();
        names.dedup();
        for n in &names {
            self.result_digest(n)?;
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let digest = files_digest(&self.run.files, &refs)?;
        Ok(self.make_dep(DepFiles::Some(names), digest, None))
    }

    /// A dependency on a named part of a file: its digest is that of the
    /// extracted bytes, and the whole file's digest is recorded.
    pub fn dep_part(&self, file: &str, part: &str, extract: impl Fn(&[u8]) -> Result<Vec<u8>>) -> Result<Dep<A>> {
        let whole = self.result_digest(file)?;
        let bytes = read_checked(self.id.key(), &self.dir, file, &whole)?;
        let digest = Key::digest_bytes(&extract(&bytes)?);
        Ok(self.make_dep(DepFiles::Part { file: file.to_string(), part: part.to_string() }, digest, Some(whole)))
    }

    /// Reads one result file, checked against its digest. For readers outside
    /// a run, such as a rendering; a running computation uses `Run::open`.
    pub fn open(&self, file: &str) -> Result<std::io::Cursor<Vec<u8>>> {
        let digest = self.result_digest(file)?;
        Ok(std::io::Cursor::new(read_checked(self.id.key(), &self.dir, file, &digest)?))
    }

    /// The provenance file names, in name order: files in the manifest that a
    /// recompute does not compare.
    pub fn provenance_files(&self) -> Vec<&str> {
        self.run.files.iter().filter(|(_, e)| e.role == Role::Provenance).map(|(n, _)| n.as_str()).collect()
    }

    /// Reads one provenance file, checked against the digest the manifest
    /// recorded for it.
    pub fn open_provenance(&self, file: &str) -> Result<std::io::Cursor<Vec<u8>>> {
        let digest = match self.run.files.get(file) {
            Some(e) if e.role == Role::Provenance => e.sha256.expect("a provenance file has a digest"),
            _ => {
                return Err(Error::BadFileName {
                    name: format!("{file} (not a provenance file of {})", self.id.key().short12()),
                })
            }
        };
        Ok(std::io::Cursor::new(read_checked(self.id.key(), &self.dir, file, &digest)?))
    }

    /// A byte copy of one stored result file.
    pub fn export(&self, file: &str, to: &ExportPath) -> Result<()> {
        let digest = self.result_digest(file)?;
        let bytes = read_checked(self.id.key(), &self.dir, file, &digest)?;
        write_file_atomic(to.path(), &bytes)
    }

    /// A value recorded in `run.json`, such as the digest of a compiled
    /// model's whole file.
    pub fn recorded(&self, name: &str) -> Option<&serde_json::Value> {
        self.run.recorded.get(name)
    }

    /// The inputs as given to the computation that made it, as `run.json`
    /// records them; provenance, never part of its identity.
    pub fn given(&self) -> &serde_json::Value {
        &self.run.given
    }

    /// When the computation that made it started.
    pub fn started(&self) -> &crate::owner::Timestamp {
        &self.run.started
    }

    /// The dependencies its identity holds, as recorded.
    pub fn deps(&self) -> &[DepInfo] {
        &self.run.deps
    }

    /// The upstreams its record names without hashing them, as recorded.
    pub fn lineage(&self) -> &[Lineage] {
        &self.run.lineage
    }
}

impl<A: Artifact> std::fmt::Debug for Stored<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stored<{}>({} at {})", A::KIND, self.id.key(), self.dir.display())
    }
}

/// The SHA-256 of the named files in name order, each written as its name (a
/// `u64` little-endian length, then the UTF-8 bytes) followed by its digest.
pub(crate) fn files_digest(files: &BTreeMap<String, FileEntry>, names: &[&str]) -> Result<Key> {
    let mut sorted: Vec<&str> = names.to_vec();
    sorted.sort_unstable();
    let mut bytes = Vec::new();
    for n in sorted {
        let Some(digest) = files.get(n).filter(|e| e.role == Role::Result).and_then(|e| e.sha256) else {
            return Err(Error::BadFileName { name: format!("{n} (not a result file)") });
        };
        bytes.extend_from_slice(&(n.len() as u64).to_le_bytes());
        bytes.extend_from_slice(n.as_bytes());
        bytes.extend_from_slice(digest.as_bytes());
    }
    Ok(Key::digest_bytes(&bytes))
}

/// Which files of an upstream a dependency names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DepFiles {
    One(String),
    Part { file: String, part: String },
    All,
    /// Several named files, in name order.
    Some(Vec<String>),
}

impl DepFiles {
    /// The `file` a dependency records: the name, `<file>:<part>`, `*`, or
    /// the names joined with `,` in name order.
    fn text(&self) -> String {
        match self {
            DepFiles::One(n) => n.clone(),
            DepFiles::Part { file, part } => format!("{file}:{part}"),
            DepFiles::All => "*".into(),
            DepFiles::Some(ns) => ns.join(","),
        }
    }

    fn parse(text: &str, whole: Option<Key>) -> DepFiles {
        if text == "*" {
            DepFiles::All
        } else if whole.is_some() {
            let (file, part) = text.split_once(':').unwrap_or((text, ""));
            DepFiles::Part { file: file.into(), part: part.into() }
        } else if text.contains(',') {
            DepFiles::Some(text.split(',').map(String::from).collect())
        } else {
            DepFiles::One(text.into())
        }
    }
}

/// One file, one named part of a file, or several result files of an
/// upstream result. Hashed as a struct with tag `dep` and version 1 over
/// `digest`, `file` and `run_id`; the upstream's kind and build, and for a
/// part the whole file's digest, are recorded and not hashed. Its record form
/// is itself.
pub struct Dep<A: Artifact> {
    id: Id<A>,
    files: DepFiles,
    digest: Key,
    build: String,
    whole: Option<Key>,
}

impl<A: Artifact> Dep<A> {
    /// The upstream's run id.
    pub fn id(&self) -> &Id<A> {
        &self.id
    }

    /// The file, `<file>:<part>`, `*`, or the names joined with `,`.
    pub fn file(&self) -> String {
        self.files.text()
    }

    pub fn digest(&self) -> Key {
        self.digest
    }

    /// The upstream's build: the key of the ambient identity it was made under.
    pub fn build(&self) -> &str {
        &self.build
    }

    pub(crate) fn files(&self) -> DepFiles {
        self.files.clone()
    }

    pub(crate) fn whole(&self) -> Option<Key> {
        self.whole
    }

    fn info(&self) -> DepInfo {
        DepInfo {
            kind: A::KIND.to_string(),
            run_id: self.id.key(),
            file: self.file(),
            sha256: self.digest,
            build: self.build.clone(),
            whole: self.whole,
        }
    }
}

impl<A: Artifact> Clone for Dep<A> {
    fn clone(&self) -> Self {
        Dep { id: self.id, files: self.files.clone(), digest: self.digest, build: self.build.clone(), whole: self.whole }
    }
}

impl<A: Artifact> std::fmt::Debug for Dep<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dep<{}>({} {} {})", A::KIND, self.id.key().short12(), self.file(), self.digest.short12())
    }
}

impl<A: Artifact> PartialEq for Dep<A> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.files == other.files && self.digest == other.digest
    }
}

impl<A: Artifact> ContentAddressed for Dep<A> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_tag("dep", 1);
        h.write_str("digest");
        h.write_key(&self.digest);
        h.write_str("file");
        h.write_str(&self.file());
        h.write_str("run_id");
        h.write_key(&self.id.key());
    }
}

impl<A: Artifact> Identity for Dep<A> {
    type Record = Dep<A>;
    fn record(&self) -> Self {
        self.clone()
    }
    fn held_content(&self) -> Vec<Key> {
        Vec::new()
    }
    fn deps(&self) -> Vec<DepInfo> {
        vec![self.info()]
    }
}

#[derive(Serialize, Deserialize)]
struct DepWire {
    run_id: Key,
    file: String,
    digest: Key,
    kind: String,
    build: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    whole: Option<Key>,
}

impl<A: Artifact> Serialize for Dep<A> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        DepWire {
            run_id: self.id.key(),
            file: self.file(),
            digest: self.digest,
            kind: A::KIND.to_string(),
            build: self.build.clone(),
            whole: self.whole,
        }
        .serialize(s)
    }
}

impl<'de, A: Artifact> Deserialize<'de> for Dep<A> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let w = DepWire::deserialize(d)?;
        if w.kind != A::KIND {
            return Err(serde::de::Error::custom(format!("a dependency on `{}` where `{}` is expected", w.kind, A::KIND)));
        }
        Ok(Dep { id: Id::from_key(w.run_id), files: DepFiles::parse(&w.file, w.whole), digest: w.digest, build: w.build, whole: w.whole })
    }
}

impl<A: Artifact> From<&Dep<A>> for Lineage {
    fn from(d: &Dep<A>) -> Lineage {
        Lineage::Result { kind: A::KIND.into(), run_id: d.id.key(), file: d.file(), digest: d.digest, build: d.build.clone() }
    }
}

/// A result in any status, for commands that show unfinished runs.
pub struct Entry<A: Artifact> {
    pub(crate) id: Id<A>,
    pub(crate) record: <A::Identity as Identity>::Record,
    pub(crate) state: EntryState,
    pub(crate) dir: PathBuf,
    pub(crate) store: Arc<Inner>,
}

/// A result's status, as a reader reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryState {
    Running { owner: Owner, liveness: Liveness },
    Completed,
    Damaged { what: String },
    /// An in-place run only.
    Failed { operation: Option<String>, reason: Option<String> },
    Interrupted { owner: Owner },
}

impl<A: Artifact> Entry<A> {
    pub fn id(&self) -> &Id<A> {
        &self.id
    }

    pub fn identity(&self) -> &<A::Identity as Identity>::Record {
        &self.record
    }

    pub fn state(&self) -> &EntryState {
        &self.state
    }

    /// The result's folder.
    pub fn folder(&self) -> &Path {
        &self.dir
    }

    /// The latest progress snapshot.
    pub fn progress(&self) -> Result<Option<serde_json::Value>> {
        let path = self.dir.join(crate::record::PROGRESS_FILE);
        match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io { op: "read", path, source: e }),
            Ok(bytes) => {
                let v: serde_json::Value = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::Parse { what: path.display().to_string(), why: e.to_string() })?;
                Ok(v.get("snapshot").cloned())
            }
        }
    }

    /// `Some` when completed and not damaged.
    pub fn stored(self) -> Option<Stored<A>> {
        match self.state {
            EntryState::Completed => Stored::open_at(&self.store, self.id.key(), &self.dir).ok(),
            _ => None,
        }
    }
}

type Cache = Mutex<HashMap<(TypeId, Key), Arc<dyn Any + Send + Sync>>>;

fn content_cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Content already parsed in this process, by digest.
pub(crate) fn cached<T: Send + Sync + 'static>(digest: &Key) -> Option<Arc<T>> {
    let c = content_cache().lock().unwrap_or_else(|e| e.into_inner());
    c.get(&(TypeId::of::<T>(), *digest)).cloned().and_then(|a| a.downcast::<T>().ok())
}

pub(crate) fn cache<T: Send + Sync + 'static>(digest: Key, value: Arc<T>) -> Arc<T> {
    let mut c = content_cache().lock().unwrap_or_else(|e| e.into_inner());
    let entry = c.entry((TypeId::of::<T>(), digest)).or_insert_with(|| value as Arc<dyn Any + Send + Sync>);
    Arc::clone(entry).downcast::<T>().expect("the cache is keyed by type")
}
