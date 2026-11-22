//! Invocation records: one per command run.
//!
//! The record is written twice: at the command's first call to the store,
//! with status `running` and the results planned so far, and when the command
//! ends, replacing the first. Between the two, outcomes are kept in memory and
//! a batch's progress goes to `<seq>.progress.json` beside it. A label is
//! checked, and the sequence number assigned, under the store-wide lock
//! `.invocations/.lock`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::artifact::{Artifact, Id};
use crate::error::{json_err, At, Error, Result};
use crate::fsutil::{try_lock, write_file_atomic, write_json_atomic, LockAttempt};
use crate::hash::Key;
use crate::inputs::Given;
use crate::owner::{this_owner, Beat, Liveness, Timestamp, STALE_AFTER};
use crate::record::{
    read_record, Counts, Failure, IndexLine, InvocationFile, InvocationStatus, Lineage, Outcome, Read, ResultEntry,
    RECORD_FORMAT,
};
use crate::storage::{Inner, Store};
use crate::stored::Stored;
use crate::Reuse;

/// The store-class inputs, merged from every source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreRequest {
    label: Option<String>,
    force: bool,
    root: Option<PathBuf>,
}

impl StoreRequest {
    pub fn new(label: Option<String>, force: bool, root: Option<PathBuf>) -> Self {
        StoreRequest { label, force, root }
    }

    /// Each value given in `self` wins over `other`'s; `force` is true if
    /// either is.
    pub fn merge(self, other: StoreRequest) -> StoreRequest {
        StoreRequest {
            label: self.label.or(other.label),
            force: self.force || other.force,
            root: self.root.or(other.root),
        }
    }

    /// The given root, else `default`, the project's.
    pub fn root(&self, default: &Path) -> PathBuf {
        self.root.clone().unwrap_or_else(|| default.to_path_buf())
    }

    /// `Force` when force was asked for.
    pub fn reuse(&self) -> Reuse {
        if self.force {
            Reuse::Force
        } else {
            Reuse::IfCompleted
        }
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// What every result of one command records: the inputs as given, the
/// execution values, the store request and the lineage. It owns the
/// invocation record.
pub struct RunInfo {
    pub(crate) shared: Arc<Shared>,
}

pub(crate) struct Shared {
    pub(crate) state: Mutex<State>,
}

pub(crate) struct State {
    pub(crate) given: Given,
    pub(crate) execution: serde_json::Value,
    pub(crate) request: StoreRequest,
    pub(crate) lineage: Vec<Lineage>,
    pub(crate) results: Vec<ResultEntry>,
    pub(crate) progress: Option<serde_json::Value>,
    started: Option<Started>,
    ended: bool,
}

struct Started {
    root: PathBuf,
    seq: u64,
    owner_started: Timestamp,
    /// Keeps this process's beat running while the invocation is open.
    _beat: Arc<Beat>,
    inner: std::sync::Weak<Inner>,
}

impl RunInfo {
    pub fn new(given: &Given, execution: &impl Serialize, request: &StoreRequest) -> Result<Self> {
        let execution = serde_json::to_value(execution)
            .map_err(|e| Error::Parse { what: "the execution inputs".into(), why: e.to_string() })?;
        Ok(RunInfo {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    given: given.clone(),
                    execution,
                    request: request.clone(),
                    lineage: Vec::new(),
                    results: Vec::new(),
                    progress: None,
                    started: None,
                    ended: false,
                }),
            }),
        })
    }

    /// Records an upstream in `run.json` without hashing it. Applies to every
    /// result this `RunInfo` is passed to from now on.
    pub fn lineage(&mut self, item: impl Into<Lineage>) {
        self.shared.lock().lineage.push(item.into());
    }

    /// Ends the invocation: writes the record's last version and returns it.
    /// A `RunInfo` dropped without this writes the last version too, with
    /// what is known.
    pub fn finish(self, store: &Store) -> Result<Invocation> {
        let file = self.shared.end()?;
        let Some(file) = file else {
            return Err(Error::NotFound { handle: "an invocation that never called the store".into() });
        };
        let took = previous_holder(&store.inner, &file)?;
        Ok(Invocation { file, took_label_from: took })
    }
}

impl Drop for RunInfo {
    fn drop(&mut self) {
        let _ = self.shared.end();
    }
}

impl Shared {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `entries` to the record and, at the command's first call to the
    /// store, writes its first version. Returns each entry's position.
    pub(crate) fn register(&self, inner: &Arc<Inner>, entries: Vec<ResultEntry>, reuse: Reuse) -> Result<Vec<usize>> {
        let mut st = self.lock();
        let first = st.results.len();
        st.results.extend(entries);
        let idx: Vec<usize> = (first..st.results.len()).collect();
        if st.started.is_none() {
            let beat = inner.beat()?;
            start(inner, &mut st, reuse, beat)?;
        }
        Ok(idx)
    }

    pub(crate) fn set(&self, idx: &[usize], outcome: Outcome, failure: Option<Failure>) {
        let mut st = self.lock();
        for &i in idx {
            if let Some(e) = st.results.get_mut(i) {
                e.outcome = outcome;
                e.failure = failure.clone();
            }
        }
    }

    pub(crate) fn seq(&self) -> Option<u64> {
        self.lock().started.as_ref().map(|s| s.seq)
    }

    /// Writes a batch's progress snapshot beside the record.
    pub(crate) fn progress(&self, snapshot: serde_json::Value) -> Result<()> {
        let mut st = self.lock();
        let Some(s) = st.started.as_ref() else { return Ok(()) };
        let path = s.root.join(".invocations").join(format!("{}.progress.json", s.seq));
        let body = serde_json::json!({ "format": RECORD_FORMAT, "at": Timestamp::now(), "snapshot": snapshot });
        write_json_atomic(&path, &body)?;
        st.progress = Some(snapshot);
        Ok(())
    }

    /// Writes the last version, once. `None` when the store was never called.
    fn end(&self) -> Result<Option<InvocationFile>> {
        let mut st = self.lock();
        if st.ended {
            return Ok(None);
        }
        st.ended = true;
        if st.started.is_none() {
            return Ok(None);
        }
        for e in st.results.iter_mut().filter(|e| e.outcome == Outcome::Planned) {
            e.outcome = Outcome::Failed;
            e.failure = Some(Failure {
                operation: None,
                reason: Some("the command ended before this result did".into()),
            });
        }
        let file = st.file(InvocationStatus::Ended);
        let s = st.started.as_ref().expect("checked above");
        write_json_atomic(&invocation_path(&s.root, s.seq), &file)?;
        let _ = std::fs::remove_file(s.root.join(".invocations").join(format!("{}.progress.json", s.seq)));
        if let Some(inner) = s.inner.upgrade() {
            inner.append_index(&IndexLine::Invocation {
                format: RECORD_FORMAT,
                seq: s.seq,
                label: file.label.clone(),
                results: file.results.iter().map(|r| r.run_id).collect(),
            })?;
        }
        Ok(Some(file))
    }
}

impl State {
    fn file(&self, status: InvocationStatus) -> InvocationFile {
        let s = self.started.as_ref().expect("an invocation file is written once started");
        let mut counts = Counts::default();
        for r in &self.results {
            match r.outcome {
                Outcome::Computed | Outcome::Reused => counts.completed += 1,
                Outcome::Failed => counts.failed += 1,
                Outcome::Planned | Outcome::Busy => {}
            }
        }
        InvocationFile {
            format: RECORD_FORMAT,
            seq: s.seq,
            record: "invocation".into(),
            status,
            owner: this_owner().clone(),
            started: s.owner_started.clone(),
            ended: (status == InvocationStatus::Ended).then(Timestamp::now),
            label: self.request.label.clone(),
            argv: self.given.argv.clone(),
            given: serde_json::Value::Object(self.given.fields.clone()),
            config_files: self.given.config_files.clone(),
            results: self.results.clone(),
            counts,
            progress: self.progress.clone(),
        }
    }
}

pub(crate) fn invocation_path(root: &Path, seq: u64) -> PathBuf {
    root.join(".invocations").join(format!("{seq}.json"))
}

/// Every invocation record under the root, by sequence number. A record that
/// cannot be read is left out.
pub(crate) fn all_invocations(root: &Path) -> Vec<InvocationFile> {
    let dir = root.join(".invocations");
    let Ok(entries) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out: Vec<InvocationFile> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let seq = name.strip_suffix(".json")?;
            seq.parse::<u64>().ok()?;
            match read_record::<InvocationFile>(&e.path()) {
                Read::Ok(f) => Some(f),
                _ => None,
            }
        })
        .collect();
    out.sort_by_key(|f| f.seq);
    out
}

/// Whether every result of an invocation completed, so it may hold its label.
pub(crate) fn all_completed(f: &InvocationFile) -> bool {
    f.status == InvocationStatus::Ended
        && f.results.iter().all(|r| matches!(r.outcome, Outcome::Computed | Outcome::Reused))
}

/// Takes the store-wide lock, checks the label, takes the next sequence
/// number and writes the record's first version, all before the lock is
/// released, so a later command's label check sees this one running.
fn start(inner: &Arc<Inner>, st: &mut State, reuse: Reuse, beat: Arc<Beat>) -> Result<()> {
    let dir = inner.root.join(".invocations");
    std::fs::create_dir_all(&dir).at("create directory", &dir)?;
    let lock_path = dir.join(".lock");
    let guard = loop {
        let live = |o: &crate::Owner| inner.liveness(o, STALE_AFTER);
        match try_lock(&lock_path, &live)? {
            LockAttempt::Acquired(g) => break g,
            LockAttempt::Held(owner, Liveness::Stale { .. }) if !owner.is_this_host() => {
                return Err(Error::InvocationLockHeld { owner, path: lock_path });
            }
            LockAttempt::Held(owner, Liveness::Unknown) if !owner.is_this_host() => {
                return Err(Error::InvocationLockHeld { owner, path: lock_path });
            }
            LockAttempt::Held(..) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    };
    let existing = all_invocations(&inner.root);
    if let Some(label) = &st.request.label {
        for f in existing.iter().filter(|f| f.label.as_deref() == Some(label.as_str())) {
            if f.status != InvocationStatus::Running || &f.owner == this_owner() {
                continue;
            }
            let l = inner.liveness(&f.owner, STALE_AFTER);
            let stopped_elsewhere = !f.owner.is_this_host() && matches!(l, Liveness::Stale { .. } | Liveness::Unknown);
            if l == Liveness::Dead || (stopped_elsewhere && reuse == Reuse::Force) {
                continue;
            }
            let running = f.results.iter().find(|r| r.outcome == Outcome::Planned).map(|r| r.run_id);
            return Err(Error::LabelInUse { label: label.clone(), running, owner: f.owner.clone() });
        }
    }
    let highest = std::fs::read_dir(&dir)
        .at("list", &dir)?
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".json")?.parse::<u64>().ok())
        .max();
    let seq = highest.map_or(1, |n| n + 1);
    beat.set_invocation(seq);
    st.started = Some(Started {
        root: inner.root.clone(),
        seq,
        owner_started: Timestamp::now(),
        _beat: beat,
        inner: Arc::downgrade(inner),
    });
    let file = st.file(InvocationStatus::Running);
    write_json_atomic(&invocation_path(&inner.root, seq), &file)?;
    drop(guard);
    Ok(())
}

/// The invocation that held `file`'s label before it, when `file` now holds it.
fn previous_holder(inner: &Inner, file: &InvocationFile) -> Result<Option<u64>> {
    let Some(label) = &file.label else { return Ok(None) };
    if !all_completed(file) {
        return Ok(None);
    }
    Ok(all_invocations(&inner.root)
        .into_iter()
        .filter(|f| f.seq < file.seq && f.label.as_deref() == Some(label.as_str()) && all_completed(f))
        .map(|f| f.seq)
        .max())
}

/// An invocation record, as read back.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub(crate) file: InvocationFile,
    pub(crate) took_label_from: Option<u64>,
}

impl Invocation {
    pub fn seq(&self) -> u64 {
        self.file.seq
    }

    pub fn label(&self) -> Option<&str> {
        self.file.label.as_deref()
    }

    /// The command line as given.
    pub fn argv(&self) -> &[String] {
        &self.file.argv
    }

    /// When this invocation now holds its label, the invocation that held it
    /// before; the project prints a note naming it.
    pub fn took_label_from(&self) -> Option<u64> {
        self.took_label_from
    }

    /// The results this invocation asked for, of kind `A`, in order, with
    /// their labels. Labels that do not read as `A::Labels` are refused.
    pub fn members<A: Artifact>(&self) -> Result<Vec<(Id<A>, A::Labels)>> {
        self.file
            .results
            .iter()
            .filter(|r| r.kind == A::KIND)
            .map(|r| {
                let labels = serde_json::from_value(r.labels.clone()).map_err(|e| Error::Parse {
                    what: format!("the labels of {} in invocation {}", r.run_id.short12(), self.file.seq),
                    why: e.to_string(),
                })?;
                Ok((Id::from_key(r.run_id), labels))
            })
            .collect()
    }

    /// Every member of kind `R::Of`, completed, with its labels.
    fn completed_members<R: Rendering>(&self, store: &Store) -> Result<Vec<Member<R::Of>>> {
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for (id, labels) in self.members::<R::Of>()? {
            match store.get(&id)? {
                Some(s) => out.push((s, labels)),
                None => missing.push(id.key()),
            }
        }
        if !missing.is_empty() {
            return Err(Error::MemberNotCompleted { invocation: self.file.seq, members: missing });
        }
        Ok(out)
    }

    /// Renders this invocation's results of kind `R::Of` to a path. Every one
    /// must be completed; otherwise [`Error::MemberNotCompleted`], naming those
    /// that are not.
    pub fn render<R: Rendering>(&self, store: &Store, options: &R::Options, to: &ExportPath) -> Result<()> {
        let members = self.completed_members::<R>(store)?;
        let mut bytes = Vec::new();
        R::render(&members, options, &mut bytes)?;
        write_file_atomic(&to.0, &bytes)
    }

    /// The same rendering, to any writer, such as the terminal.
    pub fn render_to<R: Rendering>(&self, store: &Store, options: &R::Options, out: &mut dyn Write) -> Result<()> {
        let members = self.completed_members::<R>(store)?;
        R::render(&members, options, out)
    }
}

/// One result of an invocation, completed, with its labels.
pub type Member<A> = (Stored<A>, <A as Artifact>::Labels);

impl Invocation {
    /// Every member of kind `R::Of`, completed, with the completed by-product
    /// of kind `R::By` the invocation asked for of it. A by-product entry
    /// belongs to the member entry registered before it, since a member and
    /// the by-products wanted of it are registered together.
    fn completed_members_with<R: ByProductRendering>(&self, store: &Store) -> Result<Vec<MemberWith<R::Of, R::By>>> {
        let label_err = |r: &crate::record::ResultEntry, e: serde_json::Error| Error::Parse {
            what: format!("the labels of {} in invocation {}", r.run_id.short12(), self.file.seq),
            why: e.to_string(),
        };
        let mut wanted: Vec<(Key, <R::Of as Artifact>::Labels, Option<Key>)> = Vec::new();
        for r in &self.file.results {
            if r.kind == <R::Of as Artifact>::KIND {
                wanted.push((r.run_id, serde_json::from_value(r.labels.clone()).map_err(|e| label_err(r, e))?, None));
            } else if r.kind == <R::By as Artifact>::KIND {
                if let Some(last) = wanted.last_mut() {
                    last.2 = Some(r.run_id);
                }
            }
        }
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for (key, labels, by) in wanted {
            let member = store.get(&Id::<R::Of>::from_key(key))?;
            let by_product = match by {
                Some(b) => match store.get(&Id::<R::By>::from_key(b))? {
                    Some(s) => Some(s),
                    None => {
                        missing.push(b);
                        None
                    }
                },
                None => None,
            };
            match member {
                Some(s) => out.push(((s, labels), by_product)),
                None => missing.push(key),
            }
        }
        if !missing.is_empty() {
            return Err(Error::MemberNotCompleted { invocation: self.file.seq, members: missing });
        }
        Ok(out)
    }

    /// Renders this invocation's members and their by-products to the export
    /// path: a rendering that names one file `""` writes the path itself, and
    /// any other is written under the path as a directory. A member, or an
    /// asked-for by-product, that is not completed is refused with
    /// [`Error::MemberNotCompleted`], naming each.
    pub fn render_by_products<R: ByProductRendering>(&self, store: &Store, options: &R::Options, to: &ExportPath) -> Result<()> {
        let mut files = ExportFiles::default();
        self.render_by_products_to::<R>(store, options, &mut files)?;
        files.write_to(to)
    }

    /// The same rendering, to any sink.
    pub fn render_by_products_to<R: ByProductRendering>(&self, store: &Store, options: &R::Options, out: &mut dyn ExportSink) -> Result<()> {
        let members = self.completed_members_with::<R>(store)?;
        R::render(&members, options, out)
    }
}

/// One member of an invocation with the by-product of kind `B` the
/// invocation asked for of it, if it asked for one.
pub type MemberWith<A, B> = (Member<A>, Option<Stored<B>>);

/// A rendering over an invocation's members and one by-product kind of each.
pub trait ByProductRendering {
    type Of: crate::Primary;
    type By: crate::ByProduct<Of = Self::Of>;
    type Options;
    /// Each member with the by-product the invocation asked for of it; `None`
    /// for a member the invocation asked none of.
    fn render(members: &[MemberWith<Self::Of, Self::By>], options: &Self::Options, out: &mut dyn ExportSink) -> Result<()>;
}

/// Where a rendering writes: one file, or files named under a directory.
pub trait ExportSink {
    /// The writer of `name`, relative to the export path; `""` names the
    /// export path itself. A name is opened once.
    fn file(&mut self, name: &str) -> Result<&mut dyn Write>;
}

/// An export held in memory, by file name, until it is written whole.
#[derive(Default)]
pub struct ExportFiles {
    files: Vec<(String, Vec<u8>)>,
}

impl ExportFiles {
    /// The files written, in the order they were named.
    pub fn files(&self) -> &[(String, Vec<u8>)] {
        &self.files
    }

    /// Writes the files to `to`: one file named `""` is `to` itself, any other
    /// set is written under `to` as a directory. Each file is written
    /// atomically.
    pub fn write_to(&self, to: &ExportPath) -> Result<()> {
        match self.files.as_slice() {
            [(name, bytes)] if name.is_empty() => write_file_atomic(&to.0, bytes),
            files => {
                std::fs::create_dir_all(&to.0).at("create directory", &to.0)?;
                for (name, bytes) in files {
                    let path = to.0.join(name);
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent).at("create directory", parent)?;
                    }
                    write_file_atomic(&path, bytes)?;
                }
                Ok(())
            }
        }
    }
}

impl ExportSink for ExportFiles {
    fn file(&mut self, name: &str) -> Result<&mut dyn Write> {
        let bad = name.contains("..") || Path::new(name).is_absolute() || (name.is_empty() && !self.files.is_empty());
        if bad || self.files.iter().any(|(n, _)| n == name || n.is_empty()) {
            return Err(Error::BadFileName { name: name.to_string() });
        }
        self.files.push((name.to_string(), Vec::new()));
        Ok(&mut self.files.last_mut().expect("pushed above").1)
    }
}

/// A rendering: a pure function of stored result files, an invocation's labels
/// and options.
pub trait Rendering {
    /// The kind of result it renders: the invocation's results of this kind,
    /// in order.
    type Of: Artifact;
    /// Such as date columns or a file format; built by the project from its
    /// exports.
    type Options;
    fn render(
        members: &[Member<Self::Of>],
        options: &Self::Options,
        out: &mut dyn Write,
    ) -> Result<()>;
}

/// A path the user gave for an export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPath(pub PathBuf);

impl ExportPath {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl std::str::FromStr for ExportPath {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(ExportPath(PathBuf::from(s)))
    }
}

impl Serialize for ExportPath {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

/// The digest of an invocation record's configuration files, for
/// [`Store::runs_of_configuration`].
pub(crate) fn config_digests(f: &InvocationFile) -> impl Iterator<Item = Key> + '_ {
    f.config_files.iter().map(|c| c.sha256)
}

pub(crate) fn encode_labels<T: Serialize>(labels: &T) -> Result<serde_json::Value> {
    serde_json::to_value(labels).map_err(|e| json_err("encode the labels of", Path::new("a result"), e))
}
