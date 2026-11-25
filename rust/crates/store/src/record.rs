//! The files the store writes, as their JSON forms.
//!
//! Every file carries `format`. A reader that finds another version refuses
//! the file by name, with [`crate::Error::OldRecord`], and a result whose
//! record has another version is damaged: it is never read as a result.

use std::collections::BTreeMap;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::hash::Key;
use crate::identity::DepInfo;
use crate::label::Label;
use crate::owner::{Owner, Timestamp};

/// The format version of every record the store writes except the
/// invocation record ([`INVOCATION_FORMAT`]).
pub const RECORD_FORMAT: u16 = 1;

/// The format of a store root, the `format` in `store.json`.
pub const STORE_FORMAT: u16 = 1;

pub(crate) const IDENTITY_FILE: &str = "identity.json";
pub(crate) const RUN_FILE: &str = "run.json";
pub(crate) const RUN_TMP: &str = "run.json.tmp";
pub(crate) const PROGRESS_FILE: &str = "progress.json";
pub(crate) const EVENTS_FILE: &str = "events.jsonl";
pub(crate) const STAGING_DIR: &str = ".staging";
pub(crate) const ATTEMPTS_DIR: &str = ".attempts";
pub(crate) const ATTEMPT_RECORD: &str = "attempt.json";
pub(crate) const REPRODUCE_DIR: &str = ".reproduce";
pub(crate) const LOCK_FILE: &str = ".lock";

/// How many earlier attempts an in-place folder keeps.
pub const MAX_PRESERVED_ATTEMPTS: usize = 3;

/// `store.json` at the root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct StoreFile {
    pub format: u16,
    pub top_level: String,
}

/// How a nested result relates to the folder it sits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// A by-product inside its upstream's folder.
    Upstream,
    /// A result placed with [`crate::Layout::inside`] its parent.
    Inside,
}

/// The result a nested result sits in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parent {
    pub relation: Relation,
    pub kind: String,
    pub run_id: Key,
}

/// `identity.json`: what the run id is computed from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IdentityFile {
    pub format: u16,
    pub kind: String,
    pub run_id: Key,
    pub hash_version: u16,
    /// The ambient identity's record.
    pub ambient: serde_json::Value,
    /// The identity's record.
    pub identity: serde_json::Value,
    /// The result this one sits in, for a by-product or a result placed
    /// inside a parent; `null` otherwise.
    pub parent: Option<Parent>,
}

/// A status as `run.json` states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Completed,
    Failed,
}

/// Why a run failed. Both are `None` for a claim that was dropped.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Failure {
    pub operation: Option<String>,
    pub reason: Option<String>,
}

/// A file's role in its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// In the manifest and determined by the identity.
    Result,
    /// In the manifest and not compared on a recompute.
    Provenance,
    /// Describes one attempt; outside the manifest.
    Attempt,
}

/// One file of a run, as `run.json` records it. The size, modification time
/// and digest are recorded for result and provenance files only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FileEntry {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mtime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub sha256: Option<Key>,
}

impl FileEntry {
    pub fn in_manifest(&self) -> bool {
        self.role != Role::Attempt
    }
}

/// Something a result came from that is recorded and not hashed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum Lineage {
    Result { kind: String, run_id: Key, file: String, digest: Key, build: String },
    /// A file input, by its content digest.
    Input { field: String, digest: Key },
}

/// One read of another result's file by a running computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReadEntry {
    pub kind: String,
    pub run_id: Key,
    pub file: String,
}

/// `run.json`: the status and everything recorded about the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunFile {
    pub format: u16,
    pub status: Status,
    pub owner: Owner,
    pub invocation: Option<u64>,
    pub started: Timestamp,
    pub finished: Option<Timestamp>,
    pub given: serde_json::Value,
    pub execution: serde_json::Value,
    pub files: BTreeMap<String, FileEntry>,
    pub deps: Vec<DepInfo>,
    pub lineage: Vec<Lineage>,
    pub reads: Vec<ReadEntry>,
    /// Named values a content artifact's first writer records.
    pub recorded: BTreeMap<String, serde_json::Value>,
    pub failure: Option<Failure>,
}

/// The format of an invocation record, `.invocations/<seq>.json`. Every other
/// record the store writes is [`RECORD_FORMAT`].
pub const INVOCATION_FORMAT: u16 = 2;

/// What became of one result an invocation asked for, once it ended.
///
/// Written as `"outcome": "computed" | "reused" | "busy" | "failed"`, with
/// `"failure"` beside it for a failed result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "OutcomeWire", try_from = "OutcomeWire")]
pub enum Outcome {
    Computed,
    Reused,
    /// Another owner held its claim, so this invocation neither computed nor
    /// read it.
    Busy,
    Failed(Failure),
}

impl Outcome {
    /// Whether the result is complete: computed or reused.
    pub fn completed(&self) -> bool {
        matches!(self, Outcome::Computed | Outcome::Reused)
    }
}

/// A result of a running invocation: not yet ended, or ended. Written as an
/// [`Outcome`] is, with `"outcome": "planned"` for one not yet ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "OutcomeWire", try_from = "OutcomeWire")]
pub(crate) enum Pending {
    Planned,
    Done(Outcome),
}

impl Pending {
    /// The outcome; `None` while planned.
    pub fn done(&self) -> Option<&Outcome> {
        match self {
            Pending::Planned => None,
            Pending::Done(o) => Some(o),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OutcomeTag {
    Planned,
    Computed,
    Reused,
    Busy,
    Failed,
}

/// The two fields an outcome is written as.
#[derive(Clone, Serialize, Deserialize)]
struct OutcomeWire {
    outcome: OutcomeTag,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure: Option<Failure>,
}

impl From<Outcome> for OutcomeWire {
    fn from(o: Outcome) -> Self {
        let (outcome, failure) = match o {
            Outcome::Computed => (OutcomeTag::Computed, None),
            Outcome::Reused => (OutcomeTag::Reused, None),
            Outcome::Busy => (OutcomeTag::Busy, None),
            Outcome::Failed(f) => (OutcomeTag::Failed, Some(f)),
        };
        OutcomeWire { outcome, failure }
    }
}

impl TryFrom<OutcomeWire> for Outcome {
    type Error = String;
    fn try_from(w: OutcomeWire) -> std::result::Result<Self, String> {
        match (w.outcome, w.failure) {
            (OutcomeTag::Failed, f) => Ok(Outcome::Failed(f.unwrap_or_default())),
            (OutcomeTag::Planned, _) => Err("a closed invocation record lists a result as `planned`".into()),
            (_, Some(_)) => Err("a `failure` is recorded beside an outcome that is not `failed`".into()),
            (OutcomeTag::Computed, None) => Ok(Outcome::Computed),
            (OutcomeTag::Reused, None) => Ok(Outcome::Reused),
            (OutcomeTag::Busy, None) => Ok(Outcome::Busy),
        }
    }
}

impl From<Pending> for OutcomeWire {
    fn from(p: Pending) -> Self {
        match p {
            Pending::Planned => OutcomeWire { outcome: OutcomeTag::Planned, failure: None },
            Pending::Done(o) => o.into(),
        }
    }
}

impl TryFrom<OutcomeWire> for Pending {
    type Error = String;
    fn try_from(w: OutcomeWire) -> std::result::Result<Self, String> {
        match w.outcome {
            OutcomeTag::Planned if w.failure.is_none() => Ok(Pending::Planned),
            OutcomeTag::Planned => Err("a `failure` is recorded beside a `planned` result".into()),
            _ => Outcome::try_from(w).map(Pending::Done),
        }
    }
}

/// One result an invocation asked for: its kind and run id, the attributes
/// the project gave it, and what became of it (`O`: a [`Pending`] in a
/// running record, an [`Outcome`] in a closed one).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Listing<O> {
    pub kind: String,
    pub run_id: Key,
    /// The project's names and values for the result, such as a scenario's
    /// name or the seed typed; the store never reads them.
    pub attributes: serde_json::Value,
    #[serde(flatten)]
    pub outcome: O,
}

/// A configuration file an invocation read, by the field that named it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ConfigFile {
    pub field: String,
    pub path: String,
    pub sha256: Key,
}

/// The one value of a record's `record` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RecordKind {
    Invocation,
}

/// What an invocation record fixes when the command starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Head {
    /// [`INVOCATION_FORMAT`].
    pub format: u16,
    pub record: RecordKind,
    pub seq: u64,
    pub owner: Owner,
    pub started: Timestamp,
    pub label: Option<Label>,
    pub argv: Vec<String>,
    pub given: serde_json::Value,
    pub config_files: Vec<ConfigFile>,
}

/// A running invocation's record: the results asked for so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Running {
    #[serde(flatten)]
    pub head: Head,
    pub results: Vec<Listing<Pending>>,
}

/// A closed invocation's record: when it ended, and what became of each
/// result. No result of it is planned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Closed {
    #[serde(flatten)]
    pub head: Head,
    pub ended: Timestamp,
    pub results: Vec<Listing<Outcome>>,
}

/// `.invocations/<seq>.json`: one command run, in its state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Record {
    Running(Running),
    /// The command ended and wrote its last version.
    Ended(Closed),
    /// Its owner stopped while it was running, and `clean` closed its record:
    /// a result it had not finished is recorded as failed.
    Interrupted(Closed),
}

/// One listing of a record, in either state: `outcome` is `None` for a
/// result still planned.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ListingRef<'a> {
    pub kind: &'a str,
    pub run_id: Key,
    pub attributes: &'a serde_json::Value,
    pub outcome: Option<&'a Outcome>,
}

impl ListingRef<'_> {
    /// Whether the result ended computed or reused.
    pub fn completed(&self) -> bool {
        self.outcome.is_some_and(Outcome::completed)
    }
}

impl Record {
    pub fn head(&self) -> &Head {
        match self {
            Record::Running(r) => &r.head,
            Record::Ended(c) | Record::Interrupted(c) => &c.head,
        }
    }

    pub fn seq(&self) -> u64 {
        self.head().seq
    }

    pub fn label(&self) -> Option<&str> {
        self.head().label.as_ref().map(Label::as_str)
    }

    pub fn is_running(&self) -> bool {
        matches!(self, Record::Running(_))
    }

    /// Every listing, in listed order.
    pub fn listings(&self) -> Vec<ListingRef<'_>> {
        match self {
            Record::Running(r) => r
                .results
                .iter()
                .map(|l| ListingRef { kind: &l.kind, run_id: l.run_id, attributes: &l.attributes, outcome: l.outcome.done() })
                .collect(),
            Record::Ended(c) | Record::Interrupted(c) => c
                .results
                .iter()
                .map(|l| ListingRef { kind: &l.kind, run_id: l.run_id, attributes: &l.attributes, outcome: Some(&l.outcome) })
                .collect(),
        }
    }

    /// Whether the invocation ended with every result complete, so it may
    /// hold its label.
    pub fn all_completed(&self) -> bool {
        match self {
            Record::Ended(c) => c.results.iter().all(|l| l.outcome.completed()),
            Record::Running(_) | Record::Interrupted(_) => false,
        }
    }
}

/// One line of an owner's index log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum IndexLine {
    Completed {
        format: u16,
        kind: String,
        run_id: Key,
        /// The result's folder, relative to the root.
        folder: String,
        invocation: Option<u64>,
        deps: Vec<Key>,
        lineage: Vec<Key>,
        upstream: Option<Key>,
    },
}

/// A record file read and checked for its format version.
pub(crate) enum Read<T> {
    Absent,
    /// Present and not readable as this record: unreadable, unparsable, or of
    /// another format version.
    Bad(Error),
    Ok(T),
}

/// Reads a record that carries `format`, refusing another version by name.
pub(crate) fn read_record<T: DeserializeOwned>(path: &Path) -> Read<T> {
    read_versioned(path, RECORD_FORMAT)
}

/// Reads a record whose `format` must be `expected`, refusing another
/// version by name.
pub(crate) fn read_versioned<T: DeserializeOwned>(path: &Path, expected: u16) -> Read<T> {
    let bytes = match std::fs::read(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Read::Absent,
        Err(e) => return Read::Bad(Error::Io { op: "read", path: path.to_path_buf(), source: e }),
        Ok(b) => b,
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return Read::Bad(Error::Parse { what: path.display().to_string(), why: e.to_string() }),
    };
    let found = value.get("format").and_then(|f| f.as_u64());
    if found != Some(expected as u64) {
        return Read::Bad(Error::OldRecord {
            path: path.to_path_buf(),
            found: found.map(|f| f as u16).unwrap_or(0),
            reads: expected,
        });
    }
    match serde_json::from_value(value) {
        Ok(r) => Read::Ok(r),
        Err(e) => Read::Bad(Error::Parse { what: path.display().to_string(), why: e.to_string() }),
    }
}

impl<T> Read<T> {
    pub(crate) fn required(self, path: &Path) -> Result<T> {
        match self {
            Read::Ok(t) => Ok(t),
            Read::Bad(e) => Err(e),
            Read::Absent => Err(Error::NoRecord { path: path.to_path_buf() }),
        }
    }
}
