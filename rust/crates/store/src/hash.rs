//! Keys: the one encoder every identity is hashed through, and the trait that
//! drives it.
//!
//! A [`Key`] is a 32-byte SHA-256 digest. [`ContentAddressed::content_hash`]
//! writes [`HASH_VERSION`] once, at the root, and then the value's canonical
//! bytes; nested values stream their bytes into the parent's hash through
//! [`ContentAddressed::hash_into`] and are never hashed separately first. The
//! bytes each kind of value writes are listed in the run store proposal,
//! "Keys" (`docs/dev/proposals/2026-11-20-run-store-api.md`).
//!
//! Every variable-length value is written with its length or count first, so
//! no two values concatenate to the same bytes. `f64` is not
//! `ContentAddressed`: a float enters an identity only as a [`crate::Finite`],
//! which refuses NaN and infinity and makes `-0.0` equal to `+0.0`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Version of the encoding, written first by every [`ContentAddressed::content_hash`]
/// and every run id. Raising it re-keys every stored result.
///
/// A struct or enum is tagged with the string it states, each field's name is
/// written before its bytes and fields are taken in name order, and an enum's
/// variant is written as its name.
pub const HASH_VERSION: u16 = 9;

/// A 32-byte SHA-256 digest: the key of an identity, the digest of a file's
/// bytes, or a run id. Written as 64 lowercase hex digits in every record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key([u8; 32]);

impl Key {
    /// Wraps 32 digest bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 digest bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The SHA-256 of `bytes`, with no framing: the digest a manifest records
    /// for a file holding exactly these bytes.
    pub fn digest_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// All 64 hex digits.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// The first 12 hex digits: the suffix of a group directory, a build
    /// directory and a result folder.
    pub fn short12(&self) -> String {
        hex::encode(&self.0[..6])
    }

    /// Parses 64 hex digits.
    pub fn from_hex(s: &str) -> Result<Self, KeyParseError> {
        let bytes = hex::decode(s).map_err(|_| KeyParseError::NotHex)?;
        let arr: [u8; 32] =
            bytes.as_slice().try_into().map_err(|_| KeyParseError::WrongLength(bytes.len()))?;
        Ok(Self(arr))
    }
}

/// A string that is not 64 hex digits, given where a [`Key`] was expected.
#[derive(Debug, thiserror::Error)]
pub enum KeyParseError {
    #[error("a key must be hex digits")]
    NotHex,
    #[error("a key must be 32 bytes (64 hex digits), got {0} bytes")]
    WrongLength(usize),
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for Key {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Key {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Key::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Streaming hash state. Every [`ContentAddressed`] implementation writes
/// through its typed methods, which apply the framing rules.
#[derive(Clone, Default)]
pub struct CanonicalHasher {
    state: Sha256,
}

impl CanonicalHasher {
    pub fn new() -> Self {
        Self::default()
    }

    fn write_raw(&mut self, bytes: &[u8]) {
        self.state.update(bytes);
    }

    /// A length or element count, as a `u64`, little-endian.
    pub fn write_len(&mut self, n: u64) {
        self.write_raw(&n.to_le_bytes());
    }
    pub fn write_u8(&mut self, v: u8) {
        self.write_raw(&[v]);
    }
    pub fn write_u16(&mut self, v: u16) {
        self.write_raw(&v.to_le_bytes());
    }
    pub fn write_u32(&mut self, v: u32) {
        self.write_raw(&v.to_le_bytes());
    }
    pub fn write_u64(&mut self, v: u64) {
        self.write_raw(&v.to_le_bytes());
    }
    pub fn write_i32(&mut self, v: i32) {
        self.write_raw(&v.to_le_bytes());
    }
    pub fn write_i64(&mut self, v: i64) {
        self.write_raw(&v.to_le_bytes());
    }
    pub fn write_bool(&mut self, v: bool) {
        self.write_u8(v as u8);
    }
    pub fn write_char(&mut self, v: char) {
        self.write_u32(v as u32);
    }

    /// A byte slice, preceded by its length.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_len(bytes.len() as u64);
        self.write_raw(bytes);
    }

    /// A string, preceded by its UTF-8 length.
    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    /// The start of a struct or enum: its tag as a string, then its version.
    pub fn write_tag(&mut self, tag: &str, version: u16) {
        self.write_str(tag);
        self.write_u16(version);
    }

    /// A key's 32 bytes, with no prefix.
    pub fn write_key(&mut self, key: &Key) {
        self.write_raw(&key.0);
    }

    /// The finished digest.
    pub fn finalize(self) -> Key {
        Key(self.state.finalize().into())
    }
}

/// A value with a key.
pub trait ContentAddressed {
    /// Writes this value's canonical bytes into `h`.
    fn hash_into(&self, h: &mut CanonicalHasher);

    /// SHA-256 of [`HASH_VERSION`] (`u16`, little-endian) followed by
    /// [`Self::hash_into`]'s bytes.
    fn content_hash(&self) -> Key {
        let mut h = CanonicalHasher::new();
        h.write_u16(HASH_VERSION);
        self.hash_into(&mut h);
        h.finalize()
    }
}

macro_rules! fixed_width {
    ($($t:ty => $m:ident),* $(,)?) => {$(
        impl ContentAddressed for $t {
            fn hash_into(&self, h: &mut CanonicalHasher) { h.$m(*self); }
        }
    )*};
}
fixed_width!(u8 => write_u8, u16 => write_u16, u32 => write_u32, u64 => write_u64,
             i32 => write_i32, i64 => write_i64, bool => write_bool, char => write_char);

impl ContentAddressed for usize {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        // As a u64, so a key is the same on 32- and 64-bit targets.
        h.write_u64(*self as u64);
    }
}

impl ContentAddressed for String {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_str(self);
    }
}

impl ContentAddressed for str {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_str(self);
    }
}

impl ContentAddressed for Key {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_key(self);
    }
}

impl<T: ContentAddressed + ?Sized> ContentAddressed for &T {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        (**self).hash_into(h);
    }
}

impl<T: ContentAddressed + ?Sized> ContentAddressed for Box<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        (**self).hash_into(h);
    }
}

impl<T: ContentAddressed> ContentAddressed for Option<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        match self {
            None => h.write_u8(0),
            Some(v) => {
                h.write_u8(1);
                v.hash_into(h);
            }
        }
    }
}

impl<T: ContentAddressed> ContentAddressed for Vec<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for item in self {
            item.hash_into(h);
        }
    }
}

impl<T: ContentAddressed> ContentAddressed for [T] {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for item in self {
            item.hash_into(h);
        }
    }
}

impl<K: ContentAddressed + Ord, V: ContentAddressed> ContentAddressed for std::collections::BTreeMap<K, V> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for (k, v) in self {
            k.hash_into(h);
            v.hash_into(h);
        }
    }
}

impl<T: ContentAddressed + Ord> ContentAddressed for std::collections::BTreeSet<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for item in self {
            item.hash_into(h);
        }
    }
}

impl<A: ContentAddressed, B: ContentAddressed> ContentAddressed for (A, B) {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        self.0.hash_into(h);
        self.1.hash_into(h);
    }
}
