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
use crate::label::Label;
use crate::owner::{this_owner, Beat, Liveness, Timestamp, STALE_AFTER};
use crate::record::{
    read_versioned, Closed, Failure, Head, Lineage, Listing, Outcome, Pending, Read, Record, RecordKind, Running,
    INVOCATION_FORMAT, RECORD_FORMAT,
};
use crate::storage::{Inner, Store};
use crate::stored::Stored;
use crate::Reuse;

/// The store-class inputs, merged from every source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreRequest {
    label: Option<Label>,
    force: bool,
    root: Option<PathBuf>,
}

impl StoreRequest {
    /// A request to store under `root` (`None` for the caller's default),
    /// computing each result again when `force`, with the invocation named by
    /// `label`. A label [`Label::parse`] refuses refuses the request.
    pub fn new(label: Option<&str>, force: bool, root: Option<PathBuf>) -> Result<Self> {
        let label = label.map(Label::parse).transpose()?;
        Ok(StoreRequest { label, force, root })
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
        self.label.as_ref().map(Label::as_str)
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
    pub(crate) results: Vec<Listing<Pending>>,
    started: Option<Started>,
    ended: bool,
}

struct Started {
    root: PathBuf,
    seq: u64,
    owner_started: Timestamp,
    /// Keeps this process's beat running while the invocation is open.
    _beat: Arc<Beat>,
    /// The store whose snapshot each version of the record is added to.
    inner: std::sync::Weak<Inner>,
}

impl Started {
    /// Writes `file` as the record, and adds it to the store's snapshot.
    fn write(&self, file: &Record) -> Result<()> {
        write_json_atomic(&invocation_path(&self.root, self.seq), file)?;
        if let Some(inner) = self.inner.upgrade() {
            inner.wrote_own(file);
        }
        Ok(())
    }
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
                    started: None,
                    ended: false,
                }),
            }),
        })
    }

    /// Records in this invocation that the listed result `of` was made from
    /// the stored result `source`: every listing of `of` gains the reference,
    /// which the record's last version holds. Refused with
    /// [`Error::NotListed`] unless this invocation listed `of`, and with
    /// [`Error::SourceDiffers`] when it already recorded another source for
    /// `of`: a record holds one source per result, so a reader could not tell
    /// which of two answers for which result made from `of`. The store is
    /// generic over kinds; camdl calls it with a compiled model and its model
    /// source.
    pub fn record_source<A: Artifact, B: Artifact>(&self, of: &Id<A>, source: &Id<B>) -> Result<()> {
        let reference = crate::record::Reference { kind: B::KIND.to_string(), run_id: source.key() };
        let mut st = self.shared.lock();
        let listed: Vec<usize> =
            st.results.iter().enumerate().filter(|(_, l)| l.kind == A::KIND && l.run_id == of.key()).map(|(i, _)| i).collect();
        if listed.is_empty() {
            return Err(Error::NotListed { kind: A::KIND, run_id: of.key() });
        }
        if let Some(other) = listed.iter().find_map(|&i| st.results[i].source.as_ref().filter(|r| **r != reference)) {
            return Err(Error::SourceDiffers { kind: A::KIND, run_id: of.key(), recorded: other.run_id, asked: source.key() });
        }
        for i in listed {
            st.results[i].source = Some(reference.clone());
        }
        Ok(())
    }

    /// Records an upstream in `run.json` without hashing it. Applies to every
    /// result this `RunInfo` is passed to from now on.
    pub fn lineage(&self, item: impl Into<Lineage>) {
        self.shared.lock().lineage.push(item.into());
    }

    /// Ends the invocation: writes the record's last version and returns it.
    /// A `RunInfo` dropped without this writes the last version too, with
    /// what is known.
    pub fn finish(self, store: &Store) -> Result<Invocation> {
        let Some(closed) = self.shared.end()? else {
            return Err(Error::NotFound { handle: "an invocation that never called the store".into() });
        };
        let took = previous_holder(&store.inner, &closed)?;
        Ok(Invocation { record: Record::Ended(closed), took_label_from: took })
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
    pub(crate) fn register(&self, inner: &Arc<Inner>, entries: Vec<Listing<Pending>>, reuse: Reuse) -> Result<Vec<usize>> {
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

    /// Rewrites the running record with every entry registered so far: for
    /// a command that declared its members, so the record lists them while
    /// they are computed.
    pub(crate) fn write_running(&self) -> Result<()> {
        let st = self.lock();
        let Some(s) = st.started.as_ref() else { return Ok(()) };
        if st.ended {
            return Ok(());
        }
        s.write(&Record::Running(st.running()))
    }

    pub(crate) fn set(&self, idx: &[usize], outcome: Outcome) {
        let mut st = self.lock();
        for &i in idx {
            if let Some(e) = st.results.get_mut(i) {
                e.outcome = Pending::Done(outcome.clone());
            }
        }
    }

    pub(crate) fn seq(&self) -> Option<u64> {
        self.lock().started.as_ref().map(|s| s.seq)
    }

    /// Writes a batch's progress snapshot beside the record.
    pub(crate) fn progress(&self, snapshot: serde_json::Value) -> Result<()> {
        let st = self.lock();
        let Some(s) = st.started.as_ref() else { return Ok(()) };
        let path = s.root.join(".invocations").join(format!("{}.progress.json", s.seq));
        let body = serde_json::json!({ "format": RECORD_FORMAT, "at": Timestamp::now(), "snapshot": snapshot });
        write_json_atomic(&path, &body)
    }

    /// Writes the last version, once. `None` when the store was never called.
    fn end(&self) -> Result<Option<Closed>> {
        let mut st = self.lock();
        if st.ended {
            return Ok(None);
        }
        st.ended = true;
        if st.started.is_none() {
            return Ok(None);
        }
        let closed = st.closed("the command ended before this result did");
        let s = st.started.as_ref().expect("checked above");
        s.write(&Record::Ended(closed.clone()))?;
        let _ = std::fs::remove_file(s.root.join(".invocations").join(format!("{}.progress.json", s.seq)));
        Ok(Some(closed))
    }
}

impl State {
    fn head(&self) -> Head {
        let s = self.started.as_ref().expect("an invocation record is written once started");
        Head {
            format: INVOCATION_FORMAT,
            record: RecordKind::Invocation,
            seq: s.seq,
            owner: this_owner().clone(),
            started: s.owner_started.clone(),
            label: self.request.label.clone(),
            argv: self.given.argv.clone(),
            given: serde_json::Value::Object(self.given.fields.clone()),
            config_files: self.given.config_files.clone(),
        }
    }

    fn running(&self) -> Running {
        Running { head: self.head(), results: self.results.clone() }
    }

    /// The closed record: a result still planned is recorded as failed, for
    /// `reason`.
    fn closed(&self, reason: &str) -> Closed {
        close(self.running(), reason)
    }
}

/// `running`, closed now: each result still planned is recorded as failed,
/// for `reason`.
pub(crate) fn close(running: Running, reason: &str) -> Closed {
    let results = running
        .results
        .into_iter()
        .map(|l| Listing {
            kind: l.kind,
            run_id: l.run_id,
            attributes: l.attributes,
            source: l.source,
            outcome: match l.outcome {
                Pending::Done(o) => o,
                Pending::Planned => Outcome::Failed(Failure { operation: None, reason: Some(reason.into()) }),
            },
        })
        .collect();
    Closed { head: running.head, ended: Timestamp::now(), results }
}

pub(crate) fn invocation_path(root: &Path, seq: u64) -> PathBuf {
    root.join(".invocations").join(format!("{seq}.json"))
}

/// An invocation record the store does not read: one of another format, such
/// as one written by an earlier build, or one that does not parse as a record
/// of this format. The results it lists stay reusable, since reuse goes by
/// identity; a lookup by label or by the record's listings does not see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadRecord {
    pub path: PathBuf,
    pub why: Unread,
}

/// Why an invocation record is not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unread {
    /// Its `format` is `found`; this build reads [`INVOCATION_FORMAT`].
    Format { found: u16 },
    /// It is of this format and does not read as a record of it.
    Invalid { why: String },
}

impl std::fmt::Display for UnreadRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.why {
            Unread::Format { found } => {
                write!(f, "{} has format {found}; this build reads format {INVOCATION_FORMAT}", self.path.display())
            }
            Unread::Invalid { why } => write!(f, "{} does not read: {why}", self.path.display()),
        }
    }
}

/// Every invocation record under the root, by sequence number, and the
/// records it holds that are not read.
pub(crate) fn all_invocations(root: &Path) -> (Vec<Record>, Vec<UnreadRecord>) {
    let dir = root.join(".invocations");
    let Ok(entries) = std::fs::read_dir(&dir) else { return (Vec::new(), Vec::new()) };
    let mut out = Vec::new();
    let mut unread = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(seq) = name.strip_suffix(".json") else { continue };
        if seq.parse::<u64>().is_err() {
            continue;
        }
        let path = e.path();
        match read_versioned::<Record>(&path, INVOCATION_FORMAT) {
            Read::Ok(r) => out.push(r),
            Read::Absent => {}
            Read::Bad(Error::OldRecord { found, .. }) => unread.push(UnreadRecord { path, why: Unread::Format { found } }),
            Read::Bad(e) => unread.push(UnreadRecord { path, why: Unread::Invalid { why: e.to_string() } }),
        }
    }
    out.sort_by_key(Record::seq);
    unread.sort_by(|a, b| a.path.cmp(&b.path));
    (out, unread)
}

/// Takes the store-wide lock `.invocations/.lock`, waiting while another
/// process on this host holds it. A holder on another host that is stale or
/// whose liveness is unknown is refused with [`Error::InvocationLockHeld`].
fn records_lock(inner: &Inner) -> Result<crate::fsutil::LockGuard> {
    let dir = inner.root.join(".invocations");
    std::fs::create_dir_all(&dir).at("create directory", &dir)?;
    let lock_path = dir.join(".lock");
    loop {
        let live = |o: &crate::Owner| inner.liveness(o, STALE_AFTER);
        match try_lock(&lock_path, &live)? {
            LockAttempt::Acquired(g) => return Ok(g),
            LockAttempt::Held(owner, Liveness::Stale { .. }) if !owner.is_this_host() => {
                return Err(Error::InvocationLockHeld { owner, path: lock_path });
            }
            LockAttempt::Held(owner, Liveness::Unknown) if !owner.is_this_host() => {
                return Err(Error::InvocationLockHeld { owner, path: lock_path });
            }
            LockAttempt::Held(..) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
}

/// Takes the store-wide lock, checks the label, takes the next sequence
/// number and writes the record's first version, all before the lock is
/// released, so a later command's label check sees this one running.
fn start(inner: &Arc<Inner>, st: &mut State, reuse: Reuse, beat: Arc<Beat>) -> Result<()> {
    let guard = records_lock(inner)?;
    let dir = inner.root.join(".invocations");
    if let Some(label) = st.request.label() {
        for r in inner.running().iter().filter(|r| r.head.label.as_ref().map(Label::as_str) == Some(label)) {
            if &r.head.owner == this_owner() {
                continue;
            }
            let l = inner.liveness(&r.head.owner, STALE_AFTER);
            let stopped_elsewhere = !r.head.owner.is_this_host() && matches!(l, Liveness::Stale { .. } | Liveness::Unknown);
            if l == Liveness::Dead || (stopped_elsewhere && reuse == Reuse::Force) {
                continue;
            }
            let running = r.results.iter().find(|l| l.outcome == Pending::Planned).map(|l| l.run_id);
            return Err(Error::LabelInUse { label: label.to_string(), running, owner: r.head.owner.clone() });
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
    st.started.as_ref().expect("set above").write(&Record::Running(st.running()))?;
    drop(guard);
    Ok(())
}

/// The invocation that held `closed`'s label before it, when `closed` now
/// holds it. Read from the directory under the store-wide lock, so a holder
/// another process closed a moment ago is seen.
fn previous_holder(inner: &Inner, closed: &Closed) -> Result<Option<u64>> {
    let Some(label) = &closed.head.label else { return Ok(None) };
    if !closed.results.iter().all(|l| l.outcome.completed()) {
        return Ok(None);
    }
    let guard = records_lock(inner)?;
    let held = inner
        .read_records()
        .into_iter()
        .filter(|r| r.seq() < closed.head.seq && r.label() == Some(label.as_str()) && r.all_completed())
        .map(|r| r.seq())
        .max();
    drop(guard);
    Ok(held)
}

/// One result an ended invocation listed, as a consumer outside the store
/// reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub kind: String,
    pub run_id: Key,
    /// The result this one is a by-product of.
    pub upstream: Option<Key>,
    /// The attributes the invocation record gave it.
    pub attributes: serde_json::Value,
    /// Every file of role result in the result's `run.json`, by stored name.
    pub files: std::collections::BTreeMap<String, OutputFile>,
}

/// One result file of an [`Output`]: its digest and size as `run.json`
/// records them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputFile {
    pub sha256: Key,
    pub bytes: u64,
}

/// An invocation record, as read back.
#[derive(Debug, Clone)]
pub struct Invocation {
    pub(crate) record: Record,
    pub(crate) took_label_from: Option<u64>,
}

impl Invocation {
    pub fn seq(&self) -> u64 {
        self.record.seq()
    }

    pub fn label(&self) -> Option<&str> {
        self.record.label()
    }

    /// The command line as given.
    pub fn argv(&self) -> &[String] {
        &self.record.head().argv
    }

    /// When this invocation now holds its label, the invocation that held it
    /// before; the project prints a note naming it.
    pub fn took_label_from(&self) -> Option<u64> {
        self.took_label_from
    }

    /// The results this invocation asked for, of kind `A`, in order, with
    /// their attributes. Attributes that do not read as `A::Attributes` are
    /// refused.
    pub fn members<A: Artifact>(&self) -> Result<Vec<(Id<A>, A::Attributes)>> {
        self.record
            .listings()
            .into_iter()
            .filter(|r| r.kind == A::KIND)
            .map(|r| Ok((Id::from_key(r.run_id), self.decode::<A>(r.run_id, r.attributes)?)))
            .collect()
    }

    fn decode<A: Artifact>(&self, run_id: Key, attributes: &serde_json::Value) -> Result<A::Attributes> {
        serde_json::from_value(attributes.clone()).map_err(|e| Error::Parse {
            what: format!("the attributes of {} in invocation {}", run_id.short12(), self.seq()),
            why: e.to_string(),
        })
    }

    /// Each entry this invocation listed, as its kind, run id and whether it
    /// ended `computed` or `reused`, in listed order.
    pub fn entries(&self) -> Vec<(&str, Key, bool)> {
        self.record.listings().into_iter().map(|r| (r.kind, r.run_id, r.completed())).collect()
    }

    /// The attributes this invocation gave the result `run_id` of kind
    /// `kind`, or `None` when it did not list it.
    pub fn attributes_of(&self, kind: &str, run_id: Key) -> Option<serde_json::Value> {
        self.record.listings().into_iter().find(|r| r.kind == kind && r.run_id == run_id).map(|r| r.attributes.clone())
    }

    /// Each distinct (kind, run id) this invocation listed, in first-listed
    /// order, read from the results' records alone. Refused with
    /// [`Error::MemberNotCompleted`], naming each, when any entry did not end
    /// `computed` or `reused`.
    pub fn outputs(&self, store: &Store) -> Result<Vec<Output>> {
        let listings = self.record.listings();
        let missing: Vec<Key> = listings.iter().filter(|r| !r.completed()).map(|r| r.run_id).collect();
        if !missing.is_empty() {
            return Err(Error::MemberNotCompleted { invocation: self.seq(), members: missing });
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for r in &listings {
            if !seen.insert((r.kind, r.run_id)) {
                continue;
            }
            let located = store.locate(r.run_id)?;
            if located.kind() != r.kind {
                return Err(Error::Damaged {
                    run_id: r.run_id,
                    what: format!("invocation {} lists it as `{}`, and its record says `{}`", self.seq(), r.kind, located.kind()),
                });
            }
            let files = located
                .result_files()
                .into_iter()
                .map(|(name, (sha256, bytes))| (name.to_string(), OutputFile { sha256, bytes }))
                .collect();
            out.push(Output {
                kind: r.kind.to_string(),
                run_id: r.run_id,
                upstream: located.upstream(),
                attributes: r.attributes.clone(),
                files,
            });
        }
        Ok(out)
    }

    /// Every member of kind `R::Of`, completed, with its attributes.
    fn completed_members<R: Rendering>(&self, store: &Store) -> Result<Vec<Member<R::Of>>> {
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for (id, attributes) in self.members::<R::Of>()? {
            match store.get(&id)? {
                Some(s) => out.push((s, attributes)),
                None => missing.push(id.key()),
            }
        }
        if !missing.is_empty() {
            return Err(Error::MemberNotCompleted { invocation: self.seq(), members: missing });
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

/// One result of an invocation, completed, with its attributes.
pub type Member<A> = (Stored<A>, <A as Artifact>::Attributes);

impl Invocation {
    /// Every member of kind `R::Of`, completed, with the completed by-product
    /// of kind `R::By` the invocation asked for of it. A by-product entry
    /// belongs to the member entry registered before it, since a member and
    /// the by-products wanted of it are registered together.
    fn completed_members_with<R: ByProductRendering>(&self, store: &Store) -> Result<Vec<MemberWith<R::Of, R::By>>> {
        let mut wanted: Vec<(Key, <R::Of as Artifact>::Attributes, Option<Key>)> = Vec::new();
        for r in self.record.listings() {
            if r.kind == <R::Of as Artifact>::KIND {
                wanted.push((r.run_id, self.decode::<R::Of>(r.run_id, r.attributes)?, None));
            } else if r.kind == <R::By as Artifact>::KIND {
                if let Some(last) = wanted.last_mut() {
                    last.2 = Some(r.run_id);
                }
            }
        }
        let mut out = Vec::new();
        let mut missing = Vec::new();
        for (key, attributes, by) in wanted {
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
                Some(s) => out.push(((s, attributes), by_product)),
                None => missing.push(key),
            }
        }
        if !missing.is_empty() {
            return Err(Error::MemberNotCompleted { invocation: self.seq(), members: missing });
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

/// A rendering: a pure function of stored result files, an invocation's
/// attributes and options.
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
pub(crate) fn config_digests(r: &Record) -> impl Iterator<Item = Key> + '_ {
    r.head().config_files.iter().map(|c| c.sha256)
}

pub(crate) fn encode_attributes<T: Serialize>(attributes: &T) -> Result<serde_json::Value> {
    serde_json::to_value(attributes).map_err(|e| json_err("encode the attributes of", Path::new("a result"), e))
}

/// What the project is told of the invocation records a store does not read:
/// the store's root and every such record found so far.
pub type UnreadReporter = fn(&Path, &[UnreadRecord]);

static UNREAD_REPORTER: std::sync::OnceLock<UnreadReporter> = std::sync::OnceLock::new();

/// Sets the function each store calls, once per store, the first time a read
/// of its `.invocations/` finds records it does not read. The first call
/// wins. Without one, the records are only listed by
/// [`Store::unread_records`].
pub fn report_unread_records_to(f: UnreadReporter) {
    let _ = UNREAD_REPORTER.set(f);
}

pub(crate) fn report_unread(root: &Path, unread: &[UnreadRecord]) {
    if let Some(f) = UNREAD_REPORTER.get() {
        f(root, unread);
    }
}
