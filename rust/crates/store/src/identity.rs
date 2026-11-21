//! Identities, their record forms, and content.
//!
//! An [`Identity`] is a typed value that fully determines one result's bytes.
//! Its key is [`ContentAddressed::content_hash`]. Its record form,
//! [`Identity::Record`], is what `identity.json` holds and readers get back:
//! the identity with each [`Hashed`] content replaced by its digest. The two
//! forms write the same bytes into a hash, so an identity and its record have
//! one key, and a record nested in another identity hashes as the identity it
//! came from.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result};
use crate::hash::{CanonicalHasher, ContentAddressed, Key};

/// A dependency held somewhere in an identity, as `run.json` lists it and the
/// index records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DepInfo {
    pub kind: String,
    pub run_id: Key,
    pub file: String,
    pub sha256: Key,
    pub build: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub whole: Option<Key>,
}

/// A value that can stand in an identity: it has a key and a record form.
pub trait Identity: ContentAddressed + Clone + 'static {
    type Record: ContentAddressed + Serialize + DeserializeOwned + Clone + 'static;
    fn record(&self) -> Self::Record;
    /// The digest of every [`Hashed`] this value holds, nested values included.
    fn held_content(&self) -> Vec<Key>;
    /// Every dependency this value holds, nested values included, in field
    /// order. `run.json` lists them under `deps`, and the index records them as
    /// edges.
    fn deps(&self) -> Vec<DepInfo>;
    /// `self.content_hash()`, which equals `self.record().content_hash()`.
    fn key(&self) -> Key {
        self.content_hash()
    }
}

macro_rules! own_record {
    ($($t:ty),* $(,)?) => {$(
        impl Identity for $t {
            type Record = $t;
            fn record(&self) -> $t { self.clone() }
            fn held_content(&self) -> Vec<Key> { Vec::new() }
            fn deps(&self) -> Vec<DepInfo> { Vec::new() }
        }
    )*};
}
own_record!(u8, u16, u32, u64, usize, i32, i64, bool, char, String, Key, Finite);

impl<T: Identity> Identity for Option<T> {
    type Record = Option<T::Record>;
    fn record(&self) -> Self::Record {
        self.as_ref().map(Identity::record)
    }
    fn held_content(&self) -> Vec<Key> {
        self.as_ref().map(Identity::held_content).unwrap_or_default()
    }
    fn deps(&self) -> Vec<DepInfo> {
        self.as_ref().map(Identity::deps).unwrap_or_default()
    }
}

impl<T: Identity> Identity for Vec<T> {
    type Record = Vec<T::Record>;
    fn record(&self) -> Self::Record {
        self.iter().map(Identity::record).collect()
    }
    fn held_content(&self) -> Vec<Key> {
        self.iter().flat_map(Identity::held_content).collect()
    }
    fn deps(&self) -> Vec<DepInfo> {
        self.iter().flat_map(Identity::deps).collect()
    }
}

impl<K, V> Identity for BTreeMap<K, V>
where
    K: Identity + Ord,
    K::Record: Ord,
    V: Identity,
{
    type Record = BTreeMap<K::Record, V::Record>;
    fn record(&self) -> Self::Record {
        self.iter().map(|(k, v)| (k.record(), v.record())).collect()
    }
    fn held_content(&self) -> Vec<Key> {
        self.iter().flat_map(|(k, v)| k.held_content().into_iter().chain(v.held_content())).collect()
    }
    fn deps(&self) -> Vec<DepInfo> {
        self.iter().flat_map(|(k, v)| k.deps().into_iter().chain(v.deps())).collect()
    }
}

impl<T> Identity for BTreeSet<T>
where
    T: Identity + Ord,
    T::Record: Ord,
{
    type Record = BTreeSet<T::Record>;
    fn record(&self) -> Self::Record {
        self.iter().map(Identity::record).collect()
    }
    fn held_content(&self) -> Vec<Key> {
        self.iter().flat_map(Identity::held_content).collect()
    }
    fn deps(&self) -> Vec<DepInfo> {
        self.iter().flat_map(Identity::deps).collect()
    }
}

impl<A: Identity, B: Identity> Identity for (A, B) {
    type Record = (A::Record, B::Record);
    fn record(&self) -> Self::Record {
        (self.0.record(), self.1.record())
    }
    fn held_content(&self) -> Vec<Key> {
        let mut v = self.0.held_content();
        v.extend(self.1.held_content());
        v
    }
    fn deps(&self) -> Vec<DepInfo> {
        let mut v = self.0.deps();
        v.extend(self.1.deps());
        v
    }
}

/// An `f64` that is neither NaN nor infinite, with `-0.0` made `+0.0`, so `0`
/// and `-0` are one key. Hashed as the bit pattern, a `u64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Finite(f64);

/// A NaN or infinite value given where a [`Finite`] is required.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("not a finite number: {0}")]
pub struct NotFinite(pub f64);

impl Finite {
    pub fn new(x: f64) -> std::result::Result<Self, NotFinite> {
        if !x.is_finite() {
            return Err(NotFinite(x));
        }
        Ok(Finite(if x == 0.0 { 0.0 } else { x }))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl ContentAddressed for Finite {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_u64(self.0.to_bits());
    }
}

impl Serialize for Finite {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Finite {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Finite::new(f64::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

/// A typed value with a byte form. Its digest is the SHA-256 of those bytes.
pub trait Content: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

/// Content with its digest, the SHA-256 of `to_bytes()`, computed in the only
/// constructor. Cloning shares the value, so `T` need not implement `Clone`.
pub struct Hashed<T: Content> {
    value: Arc<T>,
    digest: Key,
}

impl<T: Content> Hashed<T> {
    pub fn new(value: T) -> Self {
        let digest = Key::digest_bytes(&value.to_bytes());
        Self { value: Arc::new(value), digest }
    }

    /// Wraps a value whose bytes are already in hand, so they are encoded once.
    pub(crate) fn with_bytes(value: T, bytes: &[u8]) -> Self {
        Self { value: Arc::new(value), digest: Key::digest_bytes(bytes) }
    }

    /// The SHA-256 of the content's bytes.
    pub fn digest(&self) -> Key {
        self.digest
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub(crate) fn arc(&self) -> Arc<T> {
        Arc::clone(&self.value)
    }
}

impl<T: Content> Clone for Hashed<T> {
    fn clone(&self) -> Self {
        Self { value: Arc::clone(&self.value), digest: self.digest }
    }
}

impl<T: Content> std::fmt::Debug for Hashed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hashed({})", self.digest)
    }
}

/// Writes the digest's 32 bytes, as a [`Key`] does.
impl<T: Content> ContentAddressed for Hashed<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_key(&self.digest);
    }
}

impl<T: Content + 'static> Identity for Hashed<T> {
    type Record = Key;
    fn record(&self) -> Key {
        self.digest
    }
    fn held_content(&self) -> Vec<Key> {
        vec![self.digest]
    }
    fn deps(&self) -> Vec<DepInfo> {
        Vec::new()
    }
}

/// Raw bytes as content.
impl Content for Vec<u8> {
    fn to_bytes(&self) -> Vec<u8> {
        self.clone()
    }
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bytes.to_vec())
    }
}

/// UTF-8 text as content.
impl Content for String {
    fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        String::from_utf8(bytes.to_vec())
            .map_err(|e| Error::Parse { what: "text content".into(), why: e.to_string() })
    }
}

#[cfg(test)]
mod tests;
