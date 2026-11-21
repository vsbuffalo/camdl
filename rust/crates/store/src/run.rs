//! A claimed, running result: the files it hands out, the by-products claimed
//! with it, and how it ends.

use std::any::Any;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Write};
use std::marker::PhantomData;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::artifact::{Artifact, ByProduct, ContentArtifact, Id};
use crate::error::{At, Error, Result};
use crate::fsutil::{
    fmt_mtime, fsync_dir, quarantine, remove_all, write_json_atomic, write_json_synced, LockGuard,
};
use crate::hash::Key;
use crate::identity::Hashed;
use crate::invocation::Shared;
use crate::layout::{dir_name, Placed};
use crate::record::{
    Failure, FileEntry, IdentityFile, Outcome, ReadEntry, Role, RunFile, Status, EVENTS_FILE,
    IDENTITY_FILE, PROGRESS_FILE, RECORD_FORMAT, REPRODUCE_DIR, RUN_FILE, RUN_TMP, STAGING_DIR,
};
use crate::storage::Inner;
use crate::stored::{Dep, DepFiles, Stored};
use crate::owner::Timestamp;

/// One open file of a run.
struct OpenFile {
    file: Option<File>,
    hasher: Sha256,
    bytes: u64,
}

struct FileInner {
    name: String,
    role: Role,
    path: PathBuf,
    state: Mutex<OpenFile>,
    ended: Arc<AtomicBool>,
    /// Buffered writers alive on this file. Ending the run while one is
    /// alive is refused: its buffer would never reach the file, and the
    /// manifest would record the file without those bytes.
    writers: std::sync::atomic::AtomicUsize,
}

/// One file of a run, opened once by name. It can be sent to another thread,
/// and cloned to share it between threads; every clone writes to the same
/// file.
#[derive(Clone)]
pub struct FileHandle {
    inner: Arc<FileInner>,
}

impl FileHandle {
    /// Appends bytes to the file.
    pub fn write_all(&self, bytes: &[u8]) -> Result<()> {
        if self.inner.ended.load(Ordering::Acquire) {
            return Err(Error::WriteAfterEnd { name: self.inner.name.clone() });
        }
        let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(f) = st.file.as_mut() else { return Err(Error::WriteAfterEnd { name: self.inner.name.clone() }) };
        f.write_all(bytes).at("write", &self.inner.path)?;
        st.hasher.update(bytes);
        st.bytes += bytes.len() as u64;
        Ok(())
    }

    /// A buffered writer that appends to the file; flushed when dropped. The
    /// run cannot end while it is alive ([`Error::WriterOpen`]).
    pub fn writer(&self) -> impl Write + '_ {
        self.inner.writers.fetch_add(1, Ordering::AcqRel);
        BufferedWriter { buf: Some(std::io::BufWriter::new(HandleWriter(self))), handle: self }
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Syncs and closes the file, and returns its digest and size.
    fn close(&self) -> Result<(Key, u64)> {
        let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = st.file.take() {
            f.sync_all().at("sync", &self.inner.path)?;
        }
        let digest = Key::from_bytes(std::mem::take(&mut st.hasher).finalize().into());
        Ok((digest, st.bytes))
    }
}

struct BufferedWriter<'a> {
    buf: Option<std::io::BufWriter<HandleWriter<'a>>>,
    handle: &'a FileHandle,
}

impl Write for BufferedWriter<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.buf.as_mut().expect("present until dropped").write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.buf.as_mut().expect("present until dropped").flush()
    }
}

impl Drop for BufferedWriter<'_> {
    fn drop(&mut self) {
        // Flushed before the count falls, so a finish that sees no writer
        // sees every byte written through one.
        drop(self.buf.take());
        self.handle.inner.writers.fetch_sub(1, Ordering::AcqRel);
    }
}

struct HandleWriter<'a>(&'a FileHandle);

impl Write for HandleWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write_all(buf).map_err(std::io::Error::other)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The files one run, or one by-product's part of it, has opened.
pub(crate) struct FileTable {
    dir: PathBuf,
    reserved: Vec<&'static str>,
    files: Mutex<BTreeMap<String, FileHandle>>,
    ended: Arc<AtomicBool>,
}

impl FileTable {
    pub(crate) fn new(dir: PathBuf, kinds: Vec<&'static str>) -> Self {
        FileTable { dir, reserved: kinds, files: Mutex::new(BTreeMap::new()), ended: Arc::new(AtomicBool::new(false)) }
    }

    fn check_name(&self, name: &str) -> Result<()> {
        let bad = || Error::BadFileName { name: name.to_string() };
        let path = Path::new(name);
        if name.is_empty() || path.is_absolute() || name.contains([':', ',', '\\']) || name == "*" {
            return Err(bad());
        }
        let mut comps = Vec::new();
        for c in path.components() {
            match c {
                Component::Normal(s) => {
                    let s = s.to_string_lossy();
                    if s.starts_with('.') {
                        return Err(bad());
                    }
                    comps.push(s.into_owned());
                }
                _ => return Err(bad()),
            }
        }
        let first = comps.first().map(String::as_str).unwrap_or("");
        if comps.len() == 1 && matches!(first, IDENTITY_FILE | RUN_FILE | RUN_TMP | PROGRESS_FILE | EVENTS_FILE) {
            return Err(bad());
        }
        if self.reserved.contains(&first) {
            return Err(bad());
        }
        Ok(())
    }

    pub(crate) fn open(&self, name: &str, role: Role) -> Result<FileHandle> {
        self.check_name(name)?;
        if self.ended.load(Ordering::Acquire) {
            return Err(Error::WriteAfterEnd { name: name.to_string() });
        }
        let mut files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        if files.contains_key(name) {
            return Err(Error::FileOpenedTwice { name: name.to_string() });
        }
        let path = self.dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).at("create directory", parent)?;
        }
        let file = OpenOptions::new().create_new(true).append(true).open(&path).at("create", &path)?;
        let h = FileHandle {
            inner: Arc::new(FileInner {
                name: name.to_string(),
                role,
                path,
                state: Mutex::new(OpenFile { file: Some(file), hasher: Sha256::new(), bytes: 0 }),
                ended: Arc::clone(&self.ended),
                writers: Default::default(),
            }),
        };
        files.insert(name.to_string(), h.clone());
        Ok(h)
    }

    /// Ends every handle and returns the files as `run.json` records them.
    pub(crate) fn close_all(&self) -> Result<BTreeMap<String, FileEntry>> {
        let files = self.files.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = files.values().find(|h| h.inner.writers.load(Ordering::Acquire) > 0) {
            return Err(Error::WriterOpen { name: h.inner.name.clone() });
        }
        self.ended.store(true, Ordering::Release);
        let mut out = BTreeMap::new();
        for (name, h) in files.iter() {
            let (digest, bytes) = h.close()?;
            let entry = if h.inner.role == Role::Attempt {
                FileEntry { role: Role::Attempt, bytes: None, mtime: None, sha256: None }
            } else {
                let meta = std::fs::metadata(&h.inner.path).at("stat", &h.inner.path)?;
                FileEntry {
                    role: h.inner.role,
                    bytes: Some(bytes),
                    mtime: Some(fmt_mtime(meta.modified().at("read the modification time of", &h.inner.path)?)),
                    sha256: Some(digest),
                }
            };
            out.insert(name.clone(), entry);
        }
        Ok(out)
    }

    fn end(&self) {
        self.ended.store(true, Ordering::Release);
    }
}

/// Where a run's own files and records are.
pub(crate) enum Place {
    /// An atomic claim: `.staging/<run id>/`, placed by the layout at finish.
    Staging { dir: PathBuf, placed: Placed },
    /// An in-place run in its final folder, holding its `.lock`.
    Final { dir: PathBuf, lock: Option<LockGuard> },
    /// The upstream is stored; the run exists for its by-products, and its
    /// files go to `.reproduce/` inside the first by-product's staging folder.
    Reproducing { stored_dir: PathBuf, stored: Box<RunFile>, files_dir: PathBuf },
}

/// A by-product claimed with its upstream.
pub(crate) struct ByProductSlot {
    pub(crate) kind: &'static str,
    pub(crate) run_id: Key,
    pub(crate) label: String,
    /// Its folder while it is being produced.
    pub(crate) dir: PathBuf,
    pub(crate) entries: Vec<usize>,
    pub(crate) identity: Arc<dyn Any + Send + Sync>,
    pub(crate) table: Arc<FileTable>,
    pub(crate) run_file: RunFile,
    pub(crate) held: Vec<Key>,
    pub(crate) taken: bool,
}

/// Everything about a run that does not depend on its kind.
pub(crate) struct Core {
    pub(crate) store: Arc<Inner>,
    pub(crate) kind: &'static str,
    pub(crate) run_id: Key,
    pub(crate) place: Place,
    pub(crate) table: Arc<FileTable>,
    pub(crate) run_file: RunFile,
    pub(crate) held: Vec<Key>,
    pub(crate) info: Arc<Shared>,
    pub(crate) entries: Vec<usize>,
    pub(crate) by_products: Vec<ByProductSlot>,
    /// The folders of by-products ended by `fail_by_products` while the
    /// run's own files are written inside one of them (reproducing mode):
    /// removed when the run ends.
    pub(crate) dropped: Vec<PathBuf>,
    pub(crate) events: AtomicU64,
    pub(crate) settled: bool,
}

/// A claimed, running result. Hands out files by role; ends with `finish` or
/// `fail`. Dropped without either, it fails with no reason.
pub struct Run<A: Artifact> {
    pub(crate) core: Core,
    pub(crate) identity: A::Identity,
    pub(crate) _kind: PhantomData<fn() -> A>,
}

impl<A: Artifact> Run<A> {
    pub fn id(&self) -> Id<A> {
        Id::from_key(self.core.run_id)
    }

    pub fn identity(&self) -> &A::Identity {
        &self.identity
    }

    /// A file in the manifest, determined by the identity.
    pub fn result_file(&self, name: &str) -> Result<FileHandle> {
        self.core.table.open(name, Role::Result)
    }

    /// A file in the manifest, not compared on a recompute.
    pub fn provenance_file(&self, name: &str) -> Result<FileHandle> {
        self.core.table.open(name, Role::Provenance)
    }

    /// A file outside the manifest, describing this attempt.
    pub fn attempt_file(&self, name: &str) -> Result<FileHandle> {
        self.core.table.open(name, Role::Attempt)
    }

    /// True when `A` itself is already stored and this run exists only for
    /// by-products.
    pub fn reproducing(&self) -> bool {
        matches!(self.core.place, Place::Reproducing { .. })
    }

    /// The by-product's part of this run, if it was wanted and is not already
    /// stored. Given once.
    pub fn also<B: ByProduct<Of = A>>(&mut self) -> Result<Option<ByProductRun<B>>>
    where
        B::Identity: Send + Sync,
    {
        let Some(slot) = self.core.by_products.iter_mut().find(|s| s.kind == B::KIND && !s.taken) else {
            return Ok(None);
        };
        slot.taken = true;
        let identity = slot
            .identity
            .downcast_ref::<B::Identity>()
            .expect("a wanted by-product's identity has its kind's type")
            .clone();
        Ok(Some(ByProductRun { table: Arc::clone(&slot.table), identity, _kind: PhantomData }))
    }

    /// Replaces the latest progress snapshot atomically.
    pub fn progress(&self, snapshot: &impl Serialize) -> Result<()> {
        self.core.progress(snapshot)
    }

    /// Appends one numbered line to the event log.
    pub fn event(&self, item: &impl Serialize) -> Result<()> {
        self.core.event(item)
    }

    /// The only way a running computation reads an upstream file: the bytes
    /// are checked against the dependency's digest, and the read is recorded.
    pub fn open<U: Artifact>(&mut self, dep: &Dep<U>) -> Result<std::io::Cursor<Vec<u8>>> {
        let (dir, _) = self.core.upstream(dep)?;
        let bytes = match dep.files() {
            DepFiles::One(name) => read_checked(&dir, &name, &dep.digest())?,
            DepFiles::Part { file, .. } => {
                let whole = dep.whole().expect("a part dependency records its whole file");
                read_checked(&dir, &file, &whole)?
            }
            DepFiles::All | DepFiles::Some(_) => {
                return Err(Error::BadFileName { name: format!("{} (use open_in to read one file of it)", dep.file()) })
            }
        };
        self.core.run_file.reads.push(ReadEntry { kind: U::KIND.into(), run_id: dep.id().key(), file: dep.file() });
        Ok(std::io::Cursor::new(bytes))
    }

    /// One file out of a dependency on several files of an upstream, made
    /// with `dep_all` or `dep_files`.
    pub fn open_in<U: Artifact>(&mut self, dep: &Dep<U>, file: &str) -> Result<std::io::Cursor<Vec<u8>>> {
        let (dir, files) = self.core.upstream(dep)?;
        let names: Vec<String> = match dep.files() {
            DepFiles::All => files.iter().filter(|(_, e)| e.role == Role::Result).map(|(n, _)| n.clone()).collect(),
            DepFiles::Some(names) => names,
            _ => return Err(Error::BadFileName { name: format!("{file} (the dependency names one file; use open)") }),
        };
        if !names.iter().any(|n| n == file) {
            return Err(Error::BadFileName { name: format!("{file} (not among the dependency's files)") });
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let recomputed = crate::stored::files_digest(&files, &refs)?;
        if recomputed != dep.digest() {
            return Err(Error::Damaged {
                run_id: dep.id().key(),
                what: format!("its files no longer have the digest {} a dependency recorded", dep.digest()),
            });
        }
        let digest = files.get(file).and_then(|e| e.sha256).expect("a result file has a digest");
        let bytes = read_checked(&dir, file, &digest)?;
        self.core.run_file.reads.push(ReadEntry { kind: U::KIND.into(), run_id: dep.id().key(), file: file.into() });
        Ok(std::io::Cursor::new(bytes))
    }

    /// The typed content of a content artifact, parsed once per process and
    /// cached by digest.
    pub fn load<C: ContentArtifact>(&mut self, dep: &Dep<C>) -> Result<Arc<C::Content>> {
        if let Some(v) = crate::stored::cached::<C::Content>(&dep.digest()) {
            self.core.run_file.reads.push(ReadEntry { kind: C::KIND.into(), run_id: dep.id().key(), file: dep.file() });
            return Ok(v);
        }
        let mut r = self.open(dep)?;
        let mut bytes = Vec::new();
        r.read_to_end(&mut bytes).expect("reading a buffer cannot fail");
        let value = <C::Content as crate::Content>::from_bytes(&bytes)?;
        let hashed = Hashed::with_bytes(value, &bytes);
        Ok(crate::stored::cache::<C::Content>(dep.digest(), hashed.arc()))
    }

    /// Commits the result: the manifest, the record, and the by-products
    /// after it.
    pub fn finish(mut self) -> Result<Stored<A>> {
        let dir = self.core.finish()?;
        Stored::open_at(&self.core.store, self.core.run_id, &dir)
    }

    /// Ends the run as failed, with what was running and why.
    pub fn fail(mut self, operation: &str, reason: &str) -> Result<()> {
        self.core.fail(Some(operation.to_string()), Some(reason.to_string()))
    }

    /// Ends every by-product wanted of this run as failed, with what was
    /// running and why, and leaves the run itself to `finish` or `fail`. For
    /// a computation that completed without the state its by-products are
    /// written from. The invocation record lists each one as failed, and
    /// `finish` commits none of them, so a later request for one produces it
    /// in reproducing mode.
    pub fn fail_by_products(&mut self, operation: &str, reason: &str) {
        self.core.drop_by_products(Some(operation.to_string()), Some(reason.to_string()));
    }
}

impl<A: Artifact> Drop for Run<A> {
    fn drop(&mut self) {
        if !self.core.settled {
            let _ = self.core.fail(None, None);
        }
    }
}

/// Reads `name` in `dir` and checks it against `digest`.
pub(crate) fn read_checked(dir: &Path, name: &str, digest: &Key) -> Result<Vec<u8>> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path).at("read", &path)?;
    let found = Key::digest_bytes(&bytes);
    if &found != digest {
        return Err(Error::Damaged {
            run_id: Key::from_bytes([0; 32]),
            what: format!("{} has digest {found}, where {digest} was recorded", path.display()),
        });
    }
    Ok(bytes)
}

impl Core {
    #[cfg(test)]
    pub(crate) fn files_dir_pub(&self) -> PathBuf {
        self.files_dir().to_path_buf()
    }

    fn files_dir(&self) -> &Path {
        match &self.place {
            Place::Staging { dir, .. } | Place::Final { dir, .. } => dir,
            Place::Reproducing { files_dir, .. } => files_dir,
        }
    }

    fn progress(&self, snapshot: &impl Serialize) -> Result<()> {
        let body = serde_json::json!({ "format": RECORD_FORMAT, "at": Timestamp::now(), "snapshot": snapshot });
        let path = self.files_dir().join(PROGRESS_FILE);
        write_json_atomic(&path, &body)
    }

    fn event(&self, item: &impl Serialize) -> Result<()> {
        let n = self.events.fetch_add(1, Ordering::Relaxed) + 1;
        let line = serde_json::json!({ "format": RECORD_FORMAT, "n": n, "at": Timestamp::now(), "event": item });
        let path = self.files_dir().join(EVENTS_FILE);
        let mut f = OpenOptions::new().create(true).append(true).open(&path).at("open", &path)?;
        let mut bytes = serde_json::to_vec(&line).expect("an event line encodes");
        bytes.push(b'\n');
        f.write_all(&bytes).at("write", &path)
    }

    /// The upstream folder of a dependency, with its files as recorded.
    fn upstream<U: Artifact>(&self, dep: &Dep<U>) -> Result<(PathBuf, BTreeMap<String, FileEntry>)> {
        let found = self.store.find_completed(U::KIND, &dep.id().key())?;
        let Some((dir, run)) = found else {
            return Err(Error::NotFound { handle: dep.id().key().to_hex() });
        };
        Ok((dir, run.files))
    }

    /// The attempt files the store itself wrote, added to the record.
    fn store_attempt_files(&self, files: &mut BTreeMap<String, FileEntry>, dir: &Path) {
        for name in [PROGRESS_FILE, EVENTS_FILE] {
            if dir.join(name).is_file() {
                files.insert(name.to_string(), FileEntry { role: Role::Attempt, bytes: None, mtime: None, sha256: None });
            }
        }
    }

    pub(crate) fn check_held(held: &[Key], files: &BTreeMap<String, FileEntry>) -> Result<()> {
        for d in held {
            if !files.values().any(|e| e.role == Role::Result && e.sha256.as_ref() == Some(d)) {
                return Err(Error::ContentMissing { digest: *d });
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<PathBuf> {
        let r = self.finish_inner();
        if r.is_err() && !self.settled {
            let _ = self.fail(None, r.as_ref().err().map(|e| e.to_string()));
        }
        self.remove_dropped();
        r
    }

    /// Ends every wanted by-product as failed and takes it out of the run.
    /// A by-product's folder is removed at once, unless the run's own files
    /// are being written inside it (reproducing mode), when it is removed as
    /// the run ends.
    fn drop_by_products(&mut self, operation: Option<String>, reason: Option<String>) {
        let reproducing = matches!(self.place, Place::Reproducing { .. });
        for bp in std::mem::take(&mut self.by_products) {
            bp.table.end();
            self.store.info_set(&self.info, &bp.entries, Outcome::Failed, Some(Failure { operation: operation.clone(), reason: reason.clone() }));
            if reproducing {
                self.dropped.push(bp.dir);
            } else {
                let _ = remove_all(&bp.dir);
            }
        }
    }

    fn remove_dropped(&mut self) {
        for dir in std::mem::take(&mut self.dropped) {
            let _ = remove_all(&dir);
        }
    }

    fn finish_inner(&mut self) -> Result<PathBuf> {
        let mut files = self.table.close_all()?;
        let files_dir = self.files_dir().to_path_buf();
        self.store_attempt_files(&mut files, &files_dir);
        // A by-product is written by this run, so it records the run's reads.
        for bp in &mut self.by_products {
            bp.run_file.reads = self.run_file.reads.clone();
        }
        let mut bp_files = Vec::new();
        for bp in &self.by_products {
            let mut f = bp.table.close_all()?;
            self.store_attempt_files(&mut f, &bp.dir);
            Self::check_held(&bp.held, &f)?;
            bp_files.push(f);
        }
        let now = Timestamp::now();
        match &self.place {
            Place::Reproducing { stored_dir, stored, files_dir } => {
                let stored_dir = stored_dir.clone();
                for (name, entry) in stored.files.iter().filter(|(_, e)| e.role == Role::Result) {
                    let ours = files.get(name).and_then(|e| (e.role == Role::Result).then_some(e.sha256).flatten());
                    if ours != entry.sha256 {
                        quarantine(&self.store.root, files_dir)?;
                        let err = Error::Divergent {
                            run_id: self.run_id,
                            file: name.clone(),
                            stored: entry.sha256.unwrap_or(Key::from_bytes([0; 32])),
                            recomputed: ours.unwrap_or(Key::from_bytes([0; 32])),
                        };
                        self.fail_by_products(Some("reproducing its upstream".into()), Some(err.to_string()));
                        self.store.info_set(&self.info, &self.entries, Outcome::Reused, None);
                        self.settled = true;
                        return Err(err);
                    }
                }
                if let Some((name, _)) = files.iter().find(|(n, e)| e.role == Role::Result && !stored.files.contains_key(*n)) {
                    quarantine(&self.store.root, files_dir)?;
                    let err = Error::Divergent {
                        run_id: self.run_id,
                        file: name.clone(),
                        stored: Key::from_bytes([0; 32]),
                        recomputed: files[name].sha256.unwrap_or(Key::from_bytes([0; 32])),
                    };
                    self.fail_by_products(Some("reproducing its upstream".into()), Some(err.to_string()));
                    self.store.info_set(&self.info, &self.entries, Outcome::Reused, None);
                    self.settled = true;
                    return Err(err);
                }
                remove_all(files_dir)?;
                for (bp, f) in self.by_products.iter_mut().zip(bp_files) {
                    commit_staged_by_product(&self.store, self.run_id, &stored_dir, bp, f, &now)?;
                }
                self.store.info_set(&self.info, &self.entries, Outcome::Reused, None);
                for bp in &self.by_products {
                    self.store.info_set(&self.info, &bp.entries, Outcome::Computed, None);
                }
                self.settled = true;
                Ok(stored_dir)
            }
            Place::Staging { dir, placed } => {
                Self::check_held(&self.held, &files)?;
                let (dir, placed) = (dir.clone(), placed.clone());
                self.run_file.status = Status::Completed;
                self.run_file.finished = Some(now.clone());
                self.run_file.files = files;
                write_json_synced(&dir.join(RUN_FILE), &self.run_file)?;
                for (bp, f) in self.by_products.iter_mut().zip(bp_files) {
                    bp.run_file.status = Status::Completed;
                    bp.run_file.finished = Some(now.clone());
                    bp.run_file.files = f;
                    write_json_synced(&bp.dir.join(RUN_FILE), &bp.run_file)?;
                    fsync_dir(&bp.dir)?;
                }
                fsync_dir(&dir)?;
                let (dest, moved) = self.store.place_staged(self.kind, self.run_id, &placed, &dir)?;
                if !moved {
                    return self.met_stored_copy(&dir, &dest);
                }
                self.settled = true;
                self.store.index_completed(self.kind, self.run_id, &dest, &self.run_file, None)?;
                for bp in &self.by_products {
                    let rel = bp.dir.strip_prefix(&dir).expect("an atomic by-product is staged inside its upstream");
                    self.store.index_completed(bp.kind, bp.run_id, &dest.join(rel), &bp.run_file, Some(self.run_id))?;
                }
                self.store.info_set(&self.info, &self.entries, Outcome::Computed, None);
                for bp in &self.by_products {
                    self.store.info_set(&self.info, &bp.entries, Outcome::Computed, None);
                }
                Ok(dest)
            }
            Place::Final { dir, .. } => {
                Self::check_held(&self.held, &files)?;
                let dir = dir.clone();
                self.run_file.status = Status::Completed;
                self.run_file.finished = Some(now.clone());
                self.run_file.files = files;
                write_record_in_place(&dir, &self.run_file)?;
                if let Place::Final { lock, .. } = &mut self.place {
                    drop(lock.take());
                }
                self.settled = true;
                self.store.index_completed(self.kind, self.run_id, &dir, &self.run_file, None)?;
                self.store.info_set(&self.info, &self.entries, Outcome::Computed, None);
                for (bp, f) in self.by_products.iter_mut().zip(bp_files) {
                    commit_staged_by_product(&self.store, self.run_id, &dir, bp, f, &now)?;
                    self.store.info_set(&self.info, &bp.entries, Outcome::Computed, None);
                }
                Ok(dir)
            }
        }
    }

    /// An atomic finish that found a completed result of its own id already
    /// in place, put there by a writer whose claim this one did not see (a
    /// claim moved aside under `Force`). Same key must mean same bytes: the
    /// result files are compared, a difference is divergence and the staged
    /// files are kept in quarantine, and on a match the stored copy is the
    /// result. A by-product produced here that the stored copy lacks is
    /// recorded as failed, so a later request produces it in reproducing mode.
    fn met_stored_copy(&mut self, staging: &Path, dest: &Path) -> Result<PathBuf> {
        self.settled = true;
        let theirs: RunFile = crate::record::read_record(&dest.join(RUN_FILE)).required(dest)?;
        let ours = |n: &str| self.run_file.files.get(n).filter(|e| e.role == Role::Result).and_then(|e| e.sha256);
        let names: std::collections::BTreeSet<&String> = self
            .run_file
            .files
            .iter()
            .chain(theirs.files.iter())
            .filter(|(_, e)| e.role == Role::Result)
            .map(|(n, _)| n)
            .collect();
        for n in names {
            let t = theirs.files.get(n).filter(|e| e.role == Role::Result).and_then(|e| e.sha256);
            if ours(n) != t {
                quarantine(&self.store.root, staging)?;
                let err = Error::Divergent {
                    run_id: self.run_id,
                    file: n.clone(),
                    stored: t.unwrap_or(Key::from_bytes([0; 32])),
                    recomputed: ours(n).unwrap_or(Key::from_bytes([0; 32])),
                };
                let failure = Failure { operation: Some("storing".into()), reason: Some(err.to_string()) };
                self.store.info_set(&self.info, &self.entries, Outcome::Failed, Some(failure.clone()));
                for bp in &self.by_products {
                    self.store.info_set(&self.info, &bp.entries, Outcome::Failed, Some(failure.clone()));
                }
                return Err(err);
            }
        }
        self.store.info_set(&self.info, &self.entries, Outcome::Reused, None);
        for bp in &self.by_products {
            let stored = self.store.find_completed(bp.kind, &bp.run_id)?.is_some();
            if stored {
                self.store.info_set(&self.info, &bp.entries, Outcome::Reused, None);
            } else {
                let reason = "another process stored this result first, without this by-product".to_string();
                self.store.info_set(&self.info, &bp.entries, Outcome::Failed, Some(Failure { operation: None, reason: Some(reason) }));
            }
        }
        remove_all(staging)?;
        Ok(dest.to_path_buf())
    }

    fn fail_by_products(&mut self, operation: Option<String>, reason: Option<String>) {
        for bp in &self.by_products {
            bp.table.end();
            if !matches!(self.place, Place::Staging { .. }) {
                let _ = remove_all(&bp.dir);
            }
            self.store.info_set(&self.info, &bp.entries, Outcome::Failed, Some(Failure { operation: operation.clone(), reason: reason.clone() }));
        }
    }

    pub(crate) fn fail(&mut self, operation: Option<String>, reason: Option<String>) -> Result<()> {
        if self.settled {
            return Ok(());
        }
        self.settled = true;
        self.table.end();
        let failure = Failure { operation: operation.clone(), reason: reason.clone() };
        self.fail_by_products(operation, reason);
        match &mut self.place {
            Place::Staging { dir, .. } => {
                let r = remove_all(dir);
                self.store.info_set(&self.info, &self.entries, Outcome::Failed, Some(failure));
                r
            }
            Place::Final { dir, lock } => {
                self.run_file.status = Status::Failed;
                self.run_file.finished = Some(Timestamp::now());
                self.run_file.failure = Some(failure.clone());
                let r = write_record_in_place(dir, &self.run_file);
                drop(lock.take());
                self.store.info_set(&self.info, &self.entries, Outcome::Failed, Some(failure));
                r
            }
            Place::Reproducing { files_dir, .. } => {
                let _ = remove_all(files_dir);
                self.store.info_set(&self.info, &self.entries, Outcome::Reused, None);
                self.remove_dropped();
                Ok(())
            }
        }
    }
}

/// Renames a by-product staged in `<upstream>/.staging/` into
/// `<upstream>/<KIND>/<label>-<12 hex>/`, its record completed first.
fn commit_staged_by_product(
    store: &Arc<Inner>,
    upstream: Key,
    upstream_dir: &Path,
    bp: &mut ByProductSlot,
    files: BTreeMap<String, FileEntry>,
    now: &Timestamp,
) -> Result<()> {
    bp.run_file.status = Status::Completed;
    bp.run_file.finished = Some(now.clone());
    bp.run_file.files = files;
    write_json_synced(&bp.dir.join(RUN_FILE), &bp.run_file)?;
    fsync_dir(&bp.dir)?;
    let kind_dir = upstream_dir.join(bp.kind);
    crate::fsutil::create_dirs_durable(upstream_dir, &kind_dir)?;
    let mut dest = kind_dir.join(dir_name(&bp.label, &bp.run_id.short12()));
    if dest.exists() {
        dest = kind_dir.join(dir_name(&bp.label, &bp.run_id.to_hex()));
    }
    std::fs::rename(&bp.dir, &dest).at("move the by-product into place", &dest)?;
    fsync_dir(&kind_dir)?;
    store.index_completed(bp.kind, bp.run_id, &dest, &bp.run_file, Some(upstream))?;
    Ok(())
}

/// Replaces an in-place record through `run.json.tmp` and a rename.
pub(crate) fn write_record_in_place(dir: &Path, run: &RunFile) -> Result<()> {
    let tmp = dir.join(RUN_TMP);
    write_json_synced(&tmp, run)?;
    std::fs::rename(&tmp, dir.join(RUN_FILE)).at("replace", dir.join(RUN_FILE))?;
    fsync_dir(dir)
}

/// A by-product's files, claimed with its upstream. It has no finish or fail:
/// the upstream's run commits or fails it, and dropping it does neither.
pub struct ByProductRun<B: ByProduct> {
    table: Arc<FileTable>,
    identity: B::Identity,
    _kind: PhantomData<fn() -> B>,
}

impl<B: ByProduct> ByProductRun<B> {
    pub fn identity(&self) -> &B::Identity {
        &self.identity
    }
    pub fn result_file(&self, name: &str) -> Result<FileHandle> {
        self.table.open(name, Role::Result)
    }
    pub fn provenance_file(&self, name: &str) -> Result<FileHandle> {
        self.table.open(name, Role::Provenance)
    }
    pub fn attempt_file(&self, name: &str) -> Result<FileHandle> {
        self.table.open(name, Role::Attempt)
    }
}

/// The records of a fresh run, as written at its claim.
pub(crate) fn running_record(info: &Shared, invocation: Option<u64>, deps: Vec<crate::identity::DepInfo>) -> RunFile {
    let st = info.lock();
    RunFile {
        format: RECORD_FORMAT,
        status: Status::Running,
        owner: crate::owner::this_owner().clone(),
        invocation,
        started: Timestamp::now(),
        finished: None,
        given: serde_json::Value::Object(st.given.fields.clone()),
        execution: st.execution.clone(),
        files: BTreeMap::new(),
        deps,
        lineage: st.lineage.clone(),
        reads: Vec::new(),
        recorded: BTreeMap::new(),
        failure: None,
    }
}

/// Writes a claim's two records into `dir`.
pub(crate) fn write_claim_records(dir: &Path, ident: &IdentityFile, run: &RunFile) -> Result<()> {
    write_json_atomic(&dir.join(IDENTITY_FILE), ident)?;
    write_json_atomic(&dir.join(RUN_FILE), run)
}

pub(crate) fn staging_dir(root: &Path, run_id: &Key) -> PathBuf {
    root.join(STAGING_DIR).join(run_id.to_hex())
}

pub(crate) fn reproduce_dir(bp_dir: &Path) -> PathBuf {
    bp_dir.join(REPRODUCE_DIR)
}

#[cfg(test)]
mod tests;
