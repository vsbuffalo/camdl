//! The store's errors.

use std::path::PathBuf;

use crate::hash::Key;
use crate::owner::Owner;

/// Everything the store refuses or fails at.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A claim the caller cannot take: a live owner holds it, it is held from
    /// another host whose owner is not known to have stopped, or this process
    /// holds it.
    #[error("{0} is computing this result")]
    InProgress(Owner),
    #[error("{root} is not a store of this format: {why}")]
    NotAStore { root: PathBuf, why: String },
    #[error("a record of kind `{kind}`, which this program does not register")]
    UnknownKind { kind: String },
    #[error("the kind `{kind}` is registered twice")]
    KindRegisteredTwice { kind: &'static str },
    #[error("{root} lays results out {stored}, and this program asks for {asked}")]
    TopLevelDiffers { root: PathBuf, stored: String, asked: String },
    #[error(
        "the result {run_id} was computed again and its file `{file}` differs: stored {stored}, \
         recomputed {recomputed}. An input that changes the output is missing from its identity, \
         or the computation is not deterministic. The recomputed files are in the store's .quarantine/"
    )]
    Divergent { run_id: Key, file: String, stored: Key, recomputed: Key },
    #[error("the result {run_id} is damaged: {what}")]
    Damaged { run_id: Key, what: String },
    #[error("{path} has format version {found}; this program reads version {reads}")]
    OldRecord { path: PathBuf, found: u16, reads: u16 },
    #[error("the result {run_id} is {status}, not completed")]
    NotCompleted { run_id: Key, status: &'static str },
    #[error("no stored result matches `{handle}`")]
    NotFound { handle: String },
    #[error("`{handle}` matches {} results: {}", matches.len(), list_keys(matches))]
    Ambiguous { handle: String, matches: Vec<Key> },
    #[error("`{handle}` names a result of kind `{found}`, where `{expected}` is needed")]
    WrongKind { handle: String, expected: &'static str, found: String },
    #[error("`{handle}` names {} results, where one is needed: {}", members.len(), list_keys(members))]
    NotOneResult { handle: String, members: Vec<Key> },
    #[error("the label `{label}` names an invocation that is still running, owned by {owner}{}",
            running.map(|k| format!(", computing {k}")).unwrap_or_default())]
    LabelInUse { label: String, running: Option<Key>, owner: Owner },
    #[error("the label `{label}` is refused: {why}")]
    InvalidLabel { label: String, why: String },
    /// `RunInfo::record_source` was asked to record a source for a result
    /// this invocation did not list.
    #[error("this invocation did not list the {kind} {run_id}, so it records no source for it")]
    NotListed { kind: &'static str, run_id: Key },
    /// `RunInfo::record_source` was asked to record a second, different
    /// source for one result in one invocation.
    #[error(
        "this invocation already recorded the source {} for the {kind} {}, and is asked to record {} too; \
         a record holds one source per result",
        recorded.short12(), run_id.short12(), asked.short12()
    )]
    SourceDiffers { kind: &'static str, run_id: Key, recorded: Key, asked: Key },
    #[error(
        "{} is held by {owner}, whose beat has stopped; it may still be alive on that host. \
         Remove the lock file by hand once that process is known to have stopped",
        path.display()
    )]
    InvocationLockHeld { owner: Owner, path: PathBuf },
    #[error("invocation {invocation} has results that did not complete: {}", list_keys(members))]
    MemberNotCompleted { invocation: u64, members: Vec<Key> },
    #[error("{} holds no result record; name a run by its label, id or path", path.display())]
    NoRecord { path: PathBuf },
    #[error("the file `{name}` was opened twice in one run")]
    FileOpenedTwice { name: String },
    #[error("`{name}` is not a file name a run may use")]
    BadFileName { name: String },
    #[error("the run was ended while a buffered writer of `{name}` was still open; drop or flush it first")]
    WriterOpen { name: String },
    #[error("a write to `{name}` after its run ended")]
    WriteAfterEnd { name: String },
    /// A store copy given to [`crate::Store::import`] holds a result whose
    /// run id, or whose folder path, this store already holds with other
    /// bytes. Nothing from the copy was placed.
    #[error(
        "the result {run_id} at {} is held in this store with other bytes, at {}: two stores hold \
         different results under one identity, or one path. Nothing from the copy was placed",
        imported.display(), local.display()
    )]
    ImportedDiffers { run_id: Key, local: PathBuf, imported: PathBuf },
    /// A store copy given to [`crate::Store::import`] holds a folder that
    /// fails a check. Nothing from the copy was placed.
    #[error("{} cannot be imported: {reason}. Nothing from the copy was placed", folder.display())]
    ImportRefused { folder: PathBuf, reason: String },
    /// A run finished holding other result or provenance files than it
    /// declared with `Run::expect`: the declared files it did not write, and
    /// the files it wrote that it did not declare.
    #[error(
        "the result {run_id} holds other files than it declared: {}{}{}",
        if missing.is_empty() { String::new() } else { format!("missing {}", missing.join(", ")) },
        if missing.is_empty() || unexpected.is_empty() { "" } else { "; " },
        if unexpected.is_empty() { String::new() } else { format!("not declared {}", unexpected.join(", ")) }
    )]
    FileSetMismatch { run_id: Key, missing: Vec<String>, unexpected: Vec<String> },
    /// A run reached `finish` without declaring its files.
    #[error("the result {run_id} declared no file set before it finished; every run declares the files it stores")]
    NoExpectedFiles { run_id: Key },
    #[error("the identity holds content {digest}, and no result file has that digest")]
    ContentMissing { digest: Key },
    #[error("the content file {file} has a digest its identity does not hold")]
    ContentUnheld { file: String },
    /// An input the store reads for a project could not be parsed: a settings
    /// file, an environment variable, or a content's bytes.
    #[error("cannot read {what}: {why}")]
    Parse { what: String, why: String },
    /// A rendering refused its members, in the project's own words: what it
    /// would not write, and why.
    #[error("{0}")]
    RenderingRefused(String),
    #[error("cannot {op} {}: {source}", path.display())]
    Io { op: &'static str, path: PathBuf, source: std::io::Error },
}

fn list_keys(keys: &[Key]) -> String {
    keys.iter().map(|k| k.short12()).collect::<Vec<_>>().join(", ")
}

pub type Result<T> = std::result::Result<T, Error>;

/// What [`crate::Store::run`] returns on failure: the store's error, or the
/// computation's.
#[derive(Debug)]
pub enum RunError<E> {
    Store(Error),
    Compute(E),
}

impl<E: std::fmt::Display> std::fmt::Display for RunError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Store(e) => e.fmt(f),
            RunError::Compute(e) => e.fmt(f),
        }
    }
}

impl<E> From<Error> for RunError<E> {
    fn from(e: Error) -> Self {
        RunError::Store(e)
    }
}

/// Attaches the operation and the path to an IO error.
pub(crate) trait At<T> {
    fn at(self, op: &'static str, path: impl AsRef<std::path::Path>) -> Result<T>;
}

impl<T> At<T> for std::io::Result<T> {
    fn at(self, op: &'static str, path: impl AsRef<std::path::Path>) -> Result<T> {
        self.map_err(|source| Error::Io { op, path: path.as_ref().to_path_buf(), source })
    }
}

/// A JSON encoding or decoding failure of a record at `path`.
pub(crate) fn json_err(op: &'static str, path: &std::path::Path, e: serde_json::Error) -> Error {
    Error::Io { op, path: path.to_path_buf(), source: std::io::Error::other(e) }
}
