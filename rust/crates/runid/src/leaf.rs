//! Reading leaves: one reader of a leaf directory's `run.json`, and the one
//! walk that discovers leaves under a store subtree.
//!
//! A **leaf** is a store directory holding a `run.json` (a [`RunRecord`]).
//! [`read_leaf`] reads one directory; [`discover`] / [`discover_kind`] find
//! every leaf under a subtree. Both report a candidate whose record cannot be
//! turned into a [`Leaf`] as an [`UnreadableLeaf`] entry — a garbage or
//! truncated `run.json`, one written under a since-renamed `kind`, one this
//! process may not read, or a directory holding a `.lock` and no `run.json`
//! — never as an omission. What to do with such an entry (list it, refuse on
//! it, drop it) is the command's decision, not this module's.
//!
//! A leaf's [`LeafState`] is [`store::leaf_state`](crate::store::leaf_state):
//! the record's outcome, plus, for a `Running` record only, the `.lock`
//! evidence ([`lock_owner`]) that tells a live run from an interrupted one.
//! This module adds one guard to it: an `Interrupted` verdict is reported only
//! when a second read of `run.json` returns the same bytes (see
//! [`read_leaf`]).
//!
//! Nothing else in a leaf is read: no manifest check (that is
//! [`FsCasStore::lookup`](crate::store::FsCasStore::lookup)), no artifact.

use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::hash::ContentHash;
use crate::kind::ArtifactKind;
use crate::record::{renamed_kind_pair, renamed_kind_sentence, RunRecord};
use crate::store::{leaf_state, lock_owner, LeafState, LockOwner, ATTEMPTS_DIR};

/// One leaf: its directory, its parsed `run.json`, and the state that record
/// and its `.lock` establish.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub dir: PathBuf,
    pub record: RunRecord,
    pub state: LeafState,
}

/// A candidate leaf whose record could not be turned into a [`Leaf`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadableLeaf {
    pub dir: PathBuf,
    /// The record's `kind`, when its cheap header read; lets a kind-scoped
    /// walk exclude an unreadable record only on evidence that it is of
    /// another kind.
    pub kind: Option<ArtifactKind>,
    /// The record's `run_id`, when its cheap header read.
    pub run_id: Option<ContentHash>,
    pub reason: Unreadable,
}

/// Why a candidate leaf is an [`UnreadableLeaf`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unreadable {
    /// `run.json` exists but reading it failed (permission denied, an IO
    /// error), or the directory could not be listed and its `run.json` could
    /// not be checked.
    Io { kind: ErrorKind, message: String },
    /// `run.json` was read and is not a [`RunRecord`] (garbage, truncated,
    /// a missing field). `message` is the parser's.
    Malformed { message: String },
    /// `run.json` names a `kind` an older camdl wrote and this one has
    /// renamed ([`renamed_kind`](crate::record::renamed_kind)).
    RenamedKind { old: String, new: String },
    /// The directory holds a `.lock` and no `run.json`: a claim between
    /// taking the lock and writing its `Running` record (a live `owner`), or
    /// the debris of one that died there.
    NoRecord { owner: LockOwner },
}

impl std::fmt::Display for Unreadable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unreadable::Io { message, .. } => write!(f, "its run.json cannot be read: {message}"),
            Unreadable::Malformed { message } => {
                write!(f, "its run.json is not a run record: {message}")
            }
            Unreadable::RenamedKind { old, new } => f.write_str(&renamed_kind_sentence(old, new)),
            Unreadable::NoRecord { owner } => {
                f.write_str("it holds a .lock and no run.json")?;
                match owner {
                    LockOwner::Live { pid } => write!(f, " (the lock is held by live pid {pid})"),
                    LockOwner::Dead { pid } => {
                        write!(f, " (the lock names pid {pid}, which is not running)")
                    }
                    LockOwner::OtherHost { pid, host } => {
                        write!(f, " (the lock is held by pid {pid} on host {host})")
                    }
                    LockOwner::Unverifiable => f.write_str(" (the lock's holder cannot be read)"),
                    LockOwner::Absent => f.write_str(" (the lock has since been released)"),
                }
            }
        }
    }
}

/// What a discovery walk may exclude before the full `run.json` parse.
///
/// Every bound is one-sided: it removes only leaves the caller would discard,
/// decided from the record's cheap header (`kind`, `run_id`) or its
/// directory's mtime. An [`UnreadableLeaf`] is excluded by `kind` or `skip`
/// only when its header read and named another kind or a skipped id.
#[derive(Debug, Default, Clone, Copy)]
pub struct Scope<'a> {
    /// Keep only leaves of this kind; `None` keeps every kind.
    pub kind: Option<ArtifactKind>,
    /// Exclude leaves whose `run_id` is in this set — the per-cell members an
    /// already-discovered ensemble row represents.
    pub skip: Option<&'a HashSet<ContentHash>>,
    /// Exclude leaves whose directory mtime proves they were written before
    /// this instant (`camdl list --since`).
    pub not_before: Option<SystemTime>,
}

/// Slack allowed when treating a leaf directory's mtime as an upper bound on
/// the record's `created_at`.
///
/// The record is stamped in memory and then written into that directory, so
/// `created_at <= mtime(dir)`. A filesystem that records mtime at one-second
/// granularity can round the directory's timestamp *below* a sub-second
/// `created_at`; two seconds is far more than any such rounding and far less
/// than any `--since` a user types.
const MTIME_BOUND_SLACK: Duration = Duration::from_secs(2);

impl Scope<'_> {
    /// `true` when `dir`'s mtime proves the record it holds is older than the
    /// `--since` cutoff. Unknown mtime is never a proof — it declines to prune.
    fn excluded_by_mtime(&self, dir: &Path) -> bool {
        let Some(cutoff) = self.not_before else { return false };
        let Ok(mtime) = std::fs::metadata(dir).and_then(|m| m.modified()) else {
            return false;
        };
        mtime.checked_add(MTIME_BOUND_SLACK).is_some_and(|bound| bound < cutoff)
    }

    /// Whether reading the cheap header can rule anything out. When it cannot,
    /// reading it would be pure overhead.
    fn is_selective(&self) -> bool {
        self.kind.is_some() || self.skip.is_some_and(|m| !m.is_empty())
    }

    fn admits(&self, header: &RunHeader) -> bool {
        self.kind.is_none_or(|k| k == header.kind)
            && !self.skip.is_some_and(|m| m.contains(&header.run_id))
    }
}

/// TEST-ONLY: the closure type of [`SNAPSHOT_GAP_HOOK`].
#[cfg(test)]
type SnapshotGapHook = Box<dyn Fn(&Path)>;

#[cfg(test)]
thread_local! {
    /// Full `RunRecord` parses performed by the walk **on this thread**.
    ///
    /// The point of a scoped walk is that a leaf the caller is going to
    /// discard never costs a full parse — a claim about *work*, not about the
    /// entries returned, which are required to stay identical — so counting
    /// the parses is the only way to assert it. Thread-local because the test
    /// harness gives each test its own thread and the walk is single-threaded,
    /// so each test sees exactly its own count with no synchronization.
    /// Compiled out of the shipped binary.
    pub(crate) static FULL_PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };

    /// TEST-ONLY: run between the two reads of `run.json` that guard an
    /// `Interrupted` verdict, to stand in for a writer finalizing in that gap.
    /// Thread-local, so it fires only in the test that installs it.
    pub(crate) static SNAPSHOT_GAP_HOOK: std::cell::RefCell<Option<SnapshotGapHook>> =
        const { std::cell::RefCell::new(None) };
}

/// The two identity fields a walk reads out of a `run.json` before deciding
/// whether the record is worth materializing.
struct RunHeader {
    kind: ArtifactKind,
    run_id: ContentHash,
}

/// Read `kind` + `run_id` out of `run.json` bytes without materializing the
/// rest of the record.
///
/// `RunRecord` serializes `format_version`, `kind`, `run_id` first (its field
/// declaration order, `record`), and this stops at the second of the two — so
/// on a leaf record of several kilobytes it tokenizes roughly a hundred bytes,
/// and `levels`, `artifacts`, `output_schema`, `inputs` and `provenance` are
/// never even scanned, let alone allocated. Field order is a *speed*
/// assumption only: a record that spells those two fields last still reads
/// correctly here, just without the early exit.
///
/// `None` for anything that is not a JSON object carrying both fields —
/// inputs on which the full `RunRecord` parse also fails, so a record whose
/// header does not read is always an [`UnreadableLeaf`], never a [`Leaf`].
fn run_header(bytes: &[u8]) -> Option<RunHeader> {
    use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};

    /// The header is handed back through `out` rather than as the visitor's
    /// value because the visitor **aborts** once it has both fields, and an
    /// aborted deserialization has no value to return.
    struct HeaderVisitor<'a> {
        out: &'a mut Option<RunHeader>,
    }

    impl<'de> Visitor<'de> for HeaderVisitor<'_> {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a run.json object carrying `kind` and `run_id`")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let mut kind: Option<ArtifactKind> = None;
            let mut run_id: Option<ContentHash> = None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "kind" => kind = Some(map.next_value()?),
                    "run_id" => run_id = Some(map.next_value()?),
                    _ => {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                if let (Some(kind), Some(run_id)) = (kind, run_id) {
                    *self.out = Some(RunHeader { kind, run_id });
                    // Stop: everything after `run_id` in a leaf record is the
                    // kilobytes we are here not to read. serde_json still wants
                    // the closing brace, so the only way out mid-document is an
                    // error the caller discards.
                    return Err(A::Error::custom("header read complete"));
                }
            }
            Ok(())
        }
    }

    let mut out = None;
    let mut de = serde_json::Deserializer::from_slice(bytes);
    // The error is deliberately ignored: it is either the early-exit abort
    // (with `out` set) or a malformed record (with `out` still `None`).
    let _ = serde::Deserializer::deserialize_map(&mut de, HeaderVisitor { out: &mut out });
    out
}

/// Read the leaf at `dir`.
///
/// `Ok(None)`: `dir` is not a leaf — it holds neither a `run.json` nor a
/// `.lock`. `Err`: it is a candidate whose record could not be read.
///
/// **The interrupted-run guard.** A reader can race a writer: it reads a
/// `Running` record, the writer then commits `Completed` and releases its
/// `.lock`, and the reader finds no lock and would call a completed run
/// interrupted. So an `Interrupted` verdict stands only if a second read of
/// `run.json` returns the same bytes; if they changed, the second record is
/// classified instead.
pub fn read_leaf(dir: &Path) -> Result<Option<Leaf>, UnreadableLeaf> {
    let run_json = dir.join("run.json");
    match std::fs::metadata(&run_json) {
        Ok(m) if m.is_file() => {}
        Ok(_) => return Ok(None),
        Err(e) if is_absent(&e) => {
            return if dir.join(".lock").is_file() { Err(no_record(dir)) } else { Ok(None) };
        }
        Err(e) => return Err(io_unreadable(dir, &e)),
    }
    match read_candidate(dir, &Scope::default(), true) {
        Some(Ok(leaf)) => Ok(Some(leaf)),
        Some(Err(u)) => Err(u),
        None => Ok(None),
    }
}

/// Every candidate leaf under `under` that `scope` admits, in path order.
///
/// `under` itself is a candidate. Hidden directories (`.staging`,
/// `.quarantine`, a leaf's `.attempts`) are not descended into. A leaf's
/// declared children under `obs/` carry an `obs.json`, not a `run.json`, so
/// they are not candidates. Unreadable candidates are entries, never
/// omissions.
pub fn discover(under: &Path, scope: Scope) -> Vec<Result<Leaf, UnreadableLeaf>> {
    let mut out = discover_in_walk_order(under, scope);
    out.sort_by(|a, b| entry_dir(a).cmp(entry_dir(b)));
    out
}

/// [`discover`] over `kind`'s store partition: `root/kind.store_dir()`, with
/// `scope.kind` set to `kind`, so the partition and the kind cannot disagree.
pub fn discover_kind(
    root: &Path,
    kind: ArtifactKind,
    scope: Scope,
) -> Vec<Result<Leaf, UnreadableLeaf>> {
    discover(&root.join(kind.store_dir()), Scope { kind: Some(kind), ..scope })
}

/// [`discover`] without the final sort: the entries in the order the walk met
/// them, which depends on the filesystem's directory order.
///
/// Transitional. It exists only so the CLI's `cas_read` walk wrappers keep
/// the order they have always returned until their callers move to
/// [`discover`]; it goes when they do.
pub fn discover_in_walk_order(under: &Path, scope: Scope) -> Vec<Result<Leaf, UnreadableLeaf>> {
    let mut out = Vec::new();
    if !under.exists() {
        return out;
    }
    let mut stack = vec![under.to_path_buf()];
    while let Some(dir) = stack.pop() {
        // One `read_dir` answers both questions about this directory — which
        // children to descend into, and whether it is a candidate leaf — from
        // the entry types the directory listing already carries.
        let found = match std::fs::read_dir(&dir) {
            Ok(entries) => scan_dir(entries, &mut stack),
            // A directory that cannot be listed may still hold a readable
            // `run.json` (mode 0111); fall back to asking for it by name.
            Err(_) => match std::fs::metadata(dir.join("run.json")) {
                Ok(m) => Found { run_json: m.is_file(), lock: false },
                Err(e) if is_absent(&e) => Found::NOTHING,
                Err(e) => {
                    if !scope.excluded_by_mtime(&dir) {
                        out.push(Err(io_unreadable(&dir, &e)));
                    }
                    continue;
                }
            },
        };
        if !found.run_json && !found.lock {
            continue;
        }
        // `--since`, applied before any JSON is read.
        if scope.excluded_by_mtime(&dir) {
            continue;
        }
        if !found.run_json {
            out.push(Err(no_record(&dir)));
            continue;
        }
        if let Some(entry) = read_candidate(&dir, &scope, true) {
            out.push(entry);
        }
    }
    out
}

fn entry_dir(e: &Result<Leaf, UnreadableLeaf>) -> &Path {
    match e {
        Ok(l) => &l.dir,
        Err(u) => &u.dir,
    }
}

/// What one directory listing says about the directory.
#[derive(Clone, Copy)]
struct Found {
    run_json: bool,
    lock: bool,
}

impl Found {
    const NOTHING: Found = Found { run_json: false, lock: false };
}

/// Sort one directory listing into "descend into this" (pushed onto `stack`)
/// and "this directory holds a `run.json` / a `.lock`" (returned), reading
/// each entry's type from the listing itself.
///
/// `DirEntry::file_type` answers from the directory entry on every filesystem
/// camdl targets (APFS, ext4, XFS all populate `d_type`), where
/// `Path::is_dir`/`is_file` cost a `stat` per entry — 274k of them on the
/// synthetic 40k-leaf store, and the walk's dominant term.
///
/// A symlink is the exception: `file_type` reports the *link*, while
/// `Path::is_dir`/`is_file` report the target, and following it is the
/// behaviour a store with a symlinked subtree relies on. So a symlinked entry
/// — and only a symlinked entry — is still resolved with a `stat`.
fn scan_dir(entries: std::fs::ReadDir, stack: &mut Vec<PathBuf>) -> Found {
    let mut found = Found::NOTHING;
    for e in entries.flatten() {
        let Ok(file_type) = e.file_type() else { continue };
        let name = e.file_name();
        let is_marker = name == "run.json" || name == ".lock";
        let (is_dir, is_file) = if file_type.is_symlink() {
            let p = e.path();
            (p.is_dir(), is_marker && p.is_file())
        } else {
            (file_type.is_dir(), is_marker && file_type.is_file())
        };
        if is_dir {
            // `.staging`, `.quarantine`, and a leaf's preserved attempts
            // (`ATTEMPTS_DIR`), whose records are kept under another name
            // but whose directory must not be walked as part of the store.
            if name.to_string_lossy().starts_with('.') || name == ATTEMPTS_DIR {
                continue;
            }
            stack.push(e.path());
        } else if is_file {
            if name == "run.json" {
                found.run_json = true;
            } else {
                found.lock = true;
            }
        }
    }
    found
}

/// Read and classify `dir/run.json`, which the caller found. `None` when
/// `scope` excludes it on its header, or it vanished with no `.lock` left
/// behind (a leaf removed mid-walk). `guard`: apply the interrupted-run
/// guard ([`read_leaf`]); off for the guard's own second read.
fn read_candidate(dir: &Path, scope: &Scope, guard: bool) -> Option<Result<Leaf, UnreadableLeaf>> {
    let bytes = match std::fs::read(dir.join("run.json")) {
        Ok(b) => b,
        // Moved away between the listing and the read: a claim preserving
        // the prior attempt holds the `.lock` by then; a removal does not.
        Err(e) if is_absent(&e) => {
            return dir.join(".lock").is_file().then(|| Err(no_record(dir)));
        }
        Err(e) => return Some(Err(io_unreadable(dir, &e))),
    };
    // The cheap header decides whether this record is worth materializing.
    if scope.is_selective() {
        match run_header(&bytes) {
            Some(h) if !scope.admits(&h) => return None,
            Some(_) => {}
            // No header: the full parse would fail too. Classify it without
            // paying for one.
            None => return Some(Err(malformed(dir, &bytes, None))),
        }
    }
    #[cfg(test)]
    FULL_PARSES.with(|n| n.set(n.get() + 1));
    let record = match serde_json::from_slice::<RunRecord>(&bytes) {
        Ok(r) => r,
        Err(e) => return Some(Err(malformed(dir, &bytes, Some(e)))),
    };
    let state = leaf_state(dir, &record);
    if guard && matches!(state, LeafState::Interrupted { .. }) {
        #[cfg(test)]
        SNAPSHOT_GAP_HOOK.with(|h| {
            if let Some(h) = h.borrow().as_ref() {
                h(dir)
            }
        });
        let unchanged = std::fs::read(dir.join("run.json")).is_ok_and(|again| again == bytes);
        if !unchanged {
            return read_candidate(dir, scope, false);
        }
    }
    Some(Ok(Leaf { dir: dir.to_path_buf(), record, state }))
}

/// The [`UnreadableLeaf`] for `run.json` bytes that are not a [`RunRecord`].
/// `parse_error` is the full parse's error when one was run; otherwise the
/// header read already failed and the message comes from a value parse.
fn malformed(dir: &Path, bytes: &[u8], parse_error: Option<serde_json::Error>) -> UnreadableLeaf {
    let header = run_header(bytes);
    let reason = match renamed_kind_pair(bytes) {
        Some((old, new)) => Unreadable::RenamedKind { old: old.into(), new: new.into() },
        None => Unreadable::Malformed {
            message: match parse_error {
                Some(e) => e.to_string(),
                None => match serde_json::from_slice::<serde_json::Value>(bytes) {
                    Err(e) => e.to_string(),
                    Ok(_) => "it does not carry both `kind` and `run_id`".into(),
                },
            },
        },
    };
    UnreadableLeaf {
        dir: dir.to_path_buf(),
        kind: header.as_ref().map(|h| h.kind),
        run_id: header.map(|h| h.run_id),
        reason,
    }
}

fn io_unreadable(dir: &Path, e: &std::io::Error) -> UnreadableLeaf {
    UnreadableLeaf {
        dir: dir.to_path_buf(),
        kind: None,
        run_id: None,
        reason: Unreadable::Io { kind: e.kind(), message: e.to_string() },
    }
}

fn no_record(dir: &Path) -> UnreadableLeaf {
    UnreadableLeaf {
        dir: dir.to_path_buf(),
        kind: None,
        run_id: None,
        reason: Unreadable::NoRecord { owner: lock_owner(dir) },
    }
}

/// An error that means "there is no such file here", as opposed to "there is
/// one and it cannot be read".
fn is_absent(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory)
}

#[cfg(test)]
mod tests;
