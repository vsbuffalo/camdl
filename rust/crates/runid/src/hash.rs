//! The canonical hashing core: `ContentHash`, `CanonicalHasher`, and the
//! `ContentAddressed` trait.
//!
//! This is the load-bearing contract of the whole CAS: get the byte
//! encoding wrong and run identities are either unstable (spurious cache
//! misses) or unsound (two materially-different inputs hash equal, and the
//! second run is served the first's output — a silent wrong answer). The
//! rules below exist to make "equal *values* always produce equal bytes"
//! an invariant of the encoding, not a thing humans re-derive per type.
//!
//! One fixed 256-bit hash is pinned as the store's hash function (SHA-256;
//! `sha2` is already the only hashing dependency in the tree). The choice
//! is recorded by [`HASH_VERSION`], folded into every root hash by
//! [`ContentAddressed::content_hash`], so the function or encoding can be
//! migrated with a single bump (which invalidates the whole store — fine
//! at alpha).
//!
//! ## Encoding rules
//!
//! - **Domain separation.** Each named `RunInput` type writes, first, a
//!   stable type tag (its fully-qualified type name, length-prefixed) then
//!   its `SCHEMA_VERSION`. Two structs with coincidentally-identical field
//!   bytes cannot collide, and a per-type policy change bumps only that
//!   type. (Primitives carry no tag — the enclosing type's framing
//!   disambiguates them.)
//! - **Length-prefixing.** Every variable-length value (string, byte
//!   slice, `Vec`, map, set) writes its element count as `u64` LE before
//!   its elements. This kills the concatenation ambiguity
//!   `("ab","c") == ("a","bc")`.
//! - **Primitives.** Integers as fixed-width little-endian; `bool` as a
//!   single `0`/`1` byte; `char` as `u32`.
//! - **Floats — two intentional policies.** *Resolved user inputs* go
//!   through [`crate::FiniteF64`] (rejects `NaN`/`±Inf` at construction,
//!   normalizes `-0.0 → +0.0`). *Structural IR floats* are hashed as raw
//!   [`f64::to_bits`] via [`CanonicalHasher::write_f64_bits`],
//!   distinguishing `±0.0` and NaN payloads to match the IR's own
//!   `ConstExpr::PartialEq`. These are one hasher with a field-level
//!   policy, not two implementations — the IR-float rule lives in the
//!   hand-written impls for the `ir` tree, and `f64` is deliberately *not*
//!   `ContentAddressed` so a resolved input cannot silently pick up the
//!   wrong policy.
//! - **Maps & sets.** Iterated in sorted key order, count-prefixed.
//! - **`Option`.** Tag byte `0` = `None`; `1` then payload = `Some`.
//! - **Enums.** Variant index (`u32` LE, declaration order) then payload.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Version of the canonical hashing function + encoding. Folded into every
/// root hash by [`ContentAddressed::content_hash`] and into [`run_id`], so
/// a change to the hash function or the byte encoding migrates the whole
/// store with a single bump. Do not change this casually — it invalidates
/// every cached artifact.
///
/// [`run_id`]: crate::run_id
pub const HASH_VERSION: u16 = 1;

// ─── ContentHash ─────────────────────────────────────────────────────────────

/// A 256-bit content hash: the structural digest of a resolved input set,
/// a level slice, or an artifact's bytes. Stored as 32 raw bytes; rendered
/// as 64 lowercase hex chars in `run.json` and as an 8-char prefix in path
/// segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Wrap 32 raw digest bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32-byte digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Full 64-char lowercase hex — the form recorded in `run.json` and
    /// matched by `show`/`cat` prefix resolution.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// First 4 bytes as 8 hex chars — the `{label}-{hash8}` path segment.
    /// A path collision needs *every* level on the path to collide on its
    /// short form simultaneously; `run.json` records the full 64-char
    /// hashes for verification.
    pub fn short8(&self) -> String {
        hex::encode(&self.0[..4])
    }

    /// Digest of opaque artifact bytes — the *same* pinned hash function
    /// (SHA-256) the structural hasher uses, applied to a file's raw
    /// contents for the `run.json` manifest. This is the "never serve wrong
    /// bytes" guarantee verified at consume time; it is distinct from the
    /// structural input hash (which frames typed values), though both use
    /// SHA-256 and migrate together via [`HASH_VERSION`].
    pub fn digest_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Parse a full 64-char hex digest (the `run.json` form).
    pub fn from_hex(s: &str) -> Result<Self, HexError> {
        let bytes = hex::decode(s).map_err(|_| HexError::NotHex)?;
        let arr: [u8; 32] = bytes.as_slice().try_into().map_err(|_| HexError::WrongLen(bytes.len()))?;
        Ok(Self(arr))
    }
}

/// Error parsing a [`ContentHash`] from hex.
#[derive(Debug, thiserror::Error)]
pub enum HexError {
    #[error("content hash is not valid hex")]
    NotHex,
    #[error("content hash must be 32 bytes (64 hex chars), got {0} bytes")]
    WrongLen(usize),
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// `run.json` records hashes as 64-char hex strings, so the wire form is a
// string — not a byte array. This keeps the record human-readable.
impl Serialize for ContentHash {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ContentHash {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ContentHash::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

// ─── CanonicalHasher ─────────────────────────────────────────────────────────

/// Streaming hash state that enforces the canonical encoding. The derive
/// macro and every hand impl drive this through [`ContentAddressed`], so
/// all callers obey the same framing rules. There is exactly one of these
/// per store — building a second is the "second hashing layer" the design
/// forbids.
#[derive(Clone)]
pub struct CanonicalHasher {
    state: Sha256,
}

impl Default for CanonicalHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl CanonicalHasher {
    pub fn new() -> Self {
        Self { state: Sha256::new() }
    }

    /// Feed raw bytes straight into the digest with no framing. Internal —
    /// callers add framing (length prefixes, fixed widths) via the typed
    /// `write_*` methods.
    fn write_raw(&mut self, bytes: &[u8]) {
        self.state.update(bytes);
    }

    /// A `u64` length / count prefix.
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

    /// A length-prefixed byte slice.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_len(bytes.len() as u64);
        self.write_raw(bytes);
    }

    /// A length-prefixed (by UTF-8 byte count) string.
    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    /// The per-type domain-separation tag: the type's fully-qualified name,
    /// length-prefixed. Written first by every named-type `hash_into`.
    pub fn write_type_tag(&mut self, tag: &str) {
        self.write_str(tag);
    }

    /// The per-type schema version, written right after the type tag. A
    /// per-type policy change bumps this and re-keys only that type.
    pub fn write_schema_version(&mut self, v: u16) {
        self.write_u16(v);
    }

    /// Raw IEEE-754 bits of an `f64`, little-endian — the *structural IR
    /// float* policy. Distinguishes `+0.0`/`-0.0` and NaN payloads, matching
    /// the IR's own `ConstExpr::PartialEq` (`expr.rs`). Resolved user inputs
    /// must NOT use this directly — they go through [`crate::FiniteF64`].
    pub fn write_f64_bits(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }

    /// A count-prefixed slice of structural IR floats (e.g. `OutputSchedule::
    /// AtTimes`). Each element uses the raw-bits policy.
    pub fn write_f64_slice(&mut self, xs: &[f64]) {
        self.write_len(xs.len() as u64);
        for x in xs {
            self.write_f64_bits(*x);
        }
    }

    /// A count-prefixed `Vec`/slice of `ContentAddressed` elements, in
    /// collection order (order-sensitive — `deps` is the documented
    /// exception and sorts itself, see [`crate::Deps`]).
    pub fn write_seq<'a, T: ContentAddressed + 'a>(
        &mut self,
        items: impl ExactSizeIterator<Item = &'a T>,
    ) {
        self.write_len(items.len() as u64);
        for item in items {
            item.hash_into(self);
        }
    }

    /// A string-keyed map of `ContentAddressed` values, iterated in sorted
    /// key order and count-prefixed. Works for `HashMap` and `BTreeMap`
    /// alike — the sort makes iteration order irrelevant.
    pub fn write_str_map<'a, V: ContentAddressed + 'a>(
        &mut self,
        entries: impl Iterator<Item = (&'a String, &'a V)>,
    ) {
        let mut kv: Vec<(&String, &V)> = entries.collect();
        kv.sort_by(|a, b| a.0.cmp(b.0));
        self.write_len(kv.len() as u64);
        for (k, v) in kv {
            self.write_str(k);
            v.hash_into(self);
        }
    }

    /// A string-keyed map of structural IR floats, sorted by key and
    /// count-prefixed (e.g. `InitialConditions::Explicit`, `Preset::params`).
    pub fn write_str_f64_map<'a>(
        &mut self,
        entries: impl Iterator<Item = (&'a String, &'a f64)>,
    ) {
        let mut kv: Vec<(&String, &f64)> = entries.collect();
        kv.sort_by(|a, b| a.0.cmp(b.0));
        self.write_len(kv.len() as u64);
        for (k, v) in kv {
            self.write_str(k);
            self.write_f64_bits(*v);
        }
    }

    /// Finish and produce the 32-byte digest.
    pub fn finalize(self) -> ContentHash {
        ContentHash(self.state.finalize().into())
    }
}

// ─── ContentAddressed ────────────────────────────────────────────────────────

/// A type whose *value* has a stable structural hash. The derived
/// `#[derive(RunInput)]` macro implements this for run-input types;
/// the `ir` tree gets hand-written impls (foreign types, structural-float
/// policy). `content_hash` is the root entry point — it folds in
/// [`HASH_VERSION`] once and delegates to `hash_into`; nested fields
/// compose via `hash_into` directly (one pass, no intermediate digest).
pub trait ContentAddressed {
    /// Feed this value's canonical bytes into `h`. Composition only — does
    /// NOT fold in `HASH_VERSION` (that is the root's job).
    fn hash_into(&self, h: &mut CanonicalHasher);

    /// The root digest of this value: `hash(HASH_VERSION ++ hash_into)`.
    fn content_hash(&self) -> ContentHash {
        let mut h = CanonicalHasher::new();
        h.write_u16(HASH_VERSION);
        self.hash_into(&mut h);
        h.finalize()
    }
}

// ─── Primitive impls ─────────────────────────────────────────────────────────
//
// Primitives carry no type tag — the enclosing named type's tag + framing
// (fixed widths, count prefixes) disambiguates them. `f64` is intentionally
// absent: a raw float must pick a policy explicitly (`FiniteF64` for
// resolved inputs, `write_f64_bits` for structural IR floats).

impl ContentAddressed for u8 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_u8(*self);
    }
}
impl ContentAddressed for u16 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_u16(*self);
    }
}
impl ContentAddressed for u32 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_u32(*self);
    }
}
impl ContentAddressed for u64 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_u64(*self);
    }
}
impl ContentAddressed for usize {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        // Width-normalize so a hash is portable across 32/64-bit targets.
        h.write_u64(*self as u64);
    }
}
impl ContentAddressed for i32 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_i32(*self);
    }
}
impl ContentAddressed for i64 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_i64(*self);
    }
}
impl ContentAddressed for bool {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_bool(*self);
    }
}
impl ContentAddressed for char {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_char(*self);
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
        h.write_seq(self.iter());
    }
}

impl<T: ContentAddressed> ContentAddressed for Box<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        (**self).hash_into(h);
    }
}

// Fixed-arity tuples carry no count prefix (arity is part of the type).
// `(i32, i32)` (dimension annotations) and `(A, B)` field pairs use this.
// Note `(f64, f64)` is unavailable by design — `f64` is not `ContentAddressed`.
impl<A: ContentAddressed, B: ContentAddressed> ContentAddressed for (A, B) {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        self.0.hash_into(h);
        self.1.hash_into(h);
    }
}

// `ContentHash` itself is `ContentAddressed`: 32 fixed bytes, no length
// prefix (the width is invariant). Used when an `ArtifactRef` folds in a
// `run_id`/`digest`, and when `run_id` folds in level hashes.
impl ContentAddressed for ContentHash {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_raw(&self.0);
    }
}

// A `BTreeMap` is already iterated in sorted key order, so it hashes
// canonically with no extra sort — count-prefixed, key then value. (The
// `ir` tree's `HashMap`/`BTreeMap<String, …>` fields use the dedicated
// `write_str_map`/`write_str_f64_map` helpers because `f64` values are not
// `ContentAddressed`; this generic impl covers resolved digest types whose
// values are.)
impl<K: ContentAddressed + Ord, V: ContentAddressed> ContentAddressed
    for std::collections::BTreeMap<K, V>
{
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for (k, v) in self.iter() {
            k.hash_into(h);
            v.hash_into(h);
        }
    }
}

// A `BTreeSet` iterates in sorted order — count-prefixed, each element.
impl<T: ContentAddressed + Ord> ContentAddressed for std::collections::BTreeSet<T> {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        h.write_len(self.len() as u64);
        for item in self.iter() {
            item.hash_into(h);
        }
    }
}
