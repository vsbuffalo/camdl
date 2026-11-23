//! Kinds of stored result.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{Error, Result};
use crate::hash::{CanonicalHasher, Key, HASH_VERSION};
use crate::identity::{Content, Hashed, Identity};
use crate::layout::Placed;

/// One kind of stored result.
pub trait Artifact: 'static {
    /// Such as `"sim"`; also the kind's directory name.
    const KIND: &'static str;
    const WRITE: WriteMode;
    type Identity: Identity;

    /// What a user recognises one result by: a scenario's name, a replicate
    /// number, the seed typed. An opaque value the project supplies with each
    /// identity; the library stores it in the invocation record and never
    /// looks inside it.
    type Labels: Serialize + DeserializeOwned + Clone + 'static;
}

/// A kind that is the main output of its own computation, placed by its own
/// layout.
pub trait Primary: Artifact {
    /// Where the folder sits and what it is called. Fixed by the first writer,
    /// whose labels it may use; nothing reads it back.
    fn layout(identity: &Self::Identity, labels: &Self::Labels) -> Placed;
}

/// An artifact produced by the same computation as another.
pub trait ByProduct: Artifact {
    type Of: Primary;
    /// The upstream's identity record, which this by-product's identity holds.
    fn upstream(identity: &Self::Identity) -> &<<Self::Of as Artifact>::Identity as Identity>::Record;
    /// The label of this by-product's folder, `<label>-<12 hex of its run
    /// id>/`, which the library places inside the upstream's folder, under
    /// `KIND/`.
    fn label(identity: &Self::Identity, labels: &Self::Labels) -> String;
}

/// An artifact whose identity is its content: the digest of its one result
/// file, `FILE`. Stored with [`crate::Store::put`].
pub trait ContentArtifact: Primary<Identity = Hashed<Self::Content>> {
    type Content: Content + Send + Sync + 'static;
    /// Such as `"model.ir.json"`.
    const FILE: &'static str;
}

/// How a result's files reach its folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Written into a staging folder and renamed into place by `finish`.
    Atomic,
    /// Written in its final folder, visible while it runs.
    InPlace,
}

/// A typed run id: the hash of the kind, the ambient key and the identity key.
pub struct Id<A: Artifact> {
    key: Key,
    _kind: PhantomData<fn() -> A>,
}

impl<A: Artifact> Id<A> {
    pub fn key(&self) -> Key {
        self.key
    }

    pub(crate) fn from_key(key: Key) -> Self {
        Id { key, _kind: PhantomData }
    }

    /// The run id of `identity` under `ambient`.
    pub(crate) fn of(ambient: &Key, identity: &A::Identity) -> Self {
        Self::from_key(run_id(A::KIND, ambient, &identity.key()))
    }
}

/// SHA-256 of the hash version as a `u16`, the kind as a string, the ambient
/// key and the identity key.
pub(crate) fn run_id(kind: &str, ambient: &Key, identity: &Key) -> Key {
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_str(kind);
    h.write_key(ambient);
    h.write_key(identity);
    h.finalize()
}

impl<A: Artifact> Clone for Id<A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: Artifact> Copy for Id<A> {}
impl<A: Artifact> PartialEq for Id<A> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl<A: Artifact> Eq for Id<A> {}
impl<A: Artifact> std::fmt::Debug for Id<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Id<{}>({})", A::KIND, self.key)
    }
}
impl<A: Artifact> std::fmt::Display for Id<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.key.fmt(f)
    }
}

/// Reads one kind's identity record without its type, and returns its key.
pub(crate) type RecordCheck = fn(&serde_json::Value) -> std::result::Result<Key, String>;

pub(crate) fn record_key<I: Identity>(v: &serde_json::Value) -> std::result::Result<Key, String> {
    use crate::hash::ContentAddressed;
    serde_json::from_value::<I::Record>(v.clone()).map(|r| r.content_hash()).map_err(|e| e.to_string())
}

/// The artifact kinds a project stores. Commands that work across kinds
/// (`list`, `show`, `status`, `verify`, `reindex`) read any kind's record
/// through it.
#[derive(Clone, Default)]
pub struct Kinds {
    pub(crate) kinds: BTreeMap<&'static str, RecordCheck>,
    pub(crate) twice: Option<&'static str>,
}

impl Kinds {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `A`. A `KIND` registered twice is refused by
    /// [`crate::Store::open`] with [`Error::KindRegisteredTwice`].
    pub fn register<A: Artifact>(mut self) -> Self {
        if self.kinds.insert(A::KIND, record_key::<A::Identity>).is_some() && self.twice.is_none() {
            self.twice = Some(A::KIND);
        }
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match self.twice {
            Some(kind) => Err(Error::KindRegisteredTwice { kind }),
            None => Ok(()),
        }
    }

    pub(crate) fn has(&self, kind: &str) -> bool {
        self.kinds.contains_key(kind)
    }

    /// Reads `record` as an identity record of `kind`, and returns its key.
    pub(crate) fn record_key(&self, kind: &str, record: &serde_json::Value) -> Result<Key> {
        let check = self.kinds.get(kind).ok_or_else(|| Error::UnknownKind { kind: kind.to_string() })?;
        check(record).map_err(|why| Error::Parse { what: format!("an identity record of kind `{kind}`"), why })
    }

    /// The `KIND` of every registered artifact, in name order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.kinds.keys().copied()
    }
}

/// What sits at the top of the store's result trees. Required, with no default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopLevel {
    /// `<root>/<label>-<12 hex of the ambient key>/<kind>/...`: one directory
    /// per build. `label` is a short readable name for the build, such as a
    /// version string; it names the directory and is never hashed.
    ByBuild { label: String },
    /// `<root>/<kind>/...`: every build's results side by side.
    ByKind,
}

impl TopLevel {
    pub(crate) fn wire(&self) -> &'static str {
        match self {
            TopLevel::ByBuild { .. } => "by_build",
            TopLevel::ByKind => "by_kind",
        }
    }
}

/// The ambient identity, with whether the build could be fingerprinted.
#[derive(Debug, Clone)]
pub enum Build<T> {
    Known(T),
    /// Some part of the fingerprint could not be determined. Nothing is reused.
    Incomplete(T),
}

impl<T> Build<T> {
    pub(crate) fn parts(&self) -> (&T, bool) {
        match self {
            Build::Known(t) => (t, true),
            Build::Incomplete(t) => (t, false),
        }
    }
}
