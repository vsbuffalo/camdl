//! The encoding of the structural model hash: [`ContentHash`],
//! [`CanonicalHasher`] and the [`ContentAddressed`] trait.
//!
//! Two models hash equal exactly when every field the impls in `impls` write
//! is equal, so `camdl` can say that two compiles, or a compile and the model
//! a fit ran on, are the same model. A field an impl leaves out never tells two
//! models apart; a field written ambiguously could make two different models
//! look the same, and a check built on the hash would then pass a different
//! model. The rules below make "equal values give equal bytes, and different
//! values different bytes" a property of the encoding.
//!
//! The hash is SHA-256 of [`HASH_VERSION`] followed by the model's canonical
//! bytes ([`ContentAddressed::content_hash`]).
//!
//! ## Encoding rules
//!
//! - **Domain separation.** Each IR type writes, first, a stable type tag
//!   (its path under `ir::`, length-prefixed) then its schema version. Two
//!   types with coincidentally identical field bytes cannot collide.
//!   Primitives carry no tag; the enclosing type's framing disambiguates them.
//! - **Length-prefixing.** Every variable-length value (string, byte slice,
//!   `Vec`, map, set) writes its element count as `u64` LE before its
//!   elements, so `("ab","c")` and `("a","bc")` differ.
//! - **Primitives.** Integers as fixed-width little-endian; `bool` as a
//!   single `0`/`1` byte; `char` as `u32`.
//! - **Floats.** An IR float is hashed as its raw [`f64::to_bits`] through
//!   [`CanonicalHasher::write_f64_bits`], keeping `±0.0` and NaN payloads
//!   distinct, as the IR's own `ConstExpr::PartialEq` does. `f64` is not
//!   `ContentAddressed`, so every float states its encoding where it is
//!   written.
//! - **Maps & sets.** Iterated in sorted key order, count-prefixed.
//! - **`Option`.** Tag byte `0` = `None`; `1` then payload = `Some`.
//! - **Enums.** Variant index (`u32` LE, declaration order) then payload.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// The version folded into every model hash, ahead of the model's bytes.
/// Changing it, or any rule of the encoding, changes every model's hash and
/// so every `model_hash` camdl prints; the pins in `ir/tests` say so.
pub const HASH_VERSION: u16 = 8;

// ─── ContentHash ─────────────────────────────────────────────────────────────

/// A 256-bit SHA-256 digest: a model's structural hash. Written as 64
/// lowercase hex digits.
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

    /// All 64 lowercase hex digits.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// The first 4 bytes as 8 hex digits, for a message.
    pub fn short8(&self) -> String {
        hex::encode(&self.0[..4])
    }

    /// The SHA-256 of `bytes`, with no framing and no [`HASH_VERSION`].
    pub fn digest_bytes(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Parses 64 hex digits.
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

// Written as a string of 64 hex digits, not a byte array.
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

/// Streaming hash state that applies the encoding rules (module doc). Every
/// impl writes through its typed methods, so all of them frame values alike.
#[derive(Clone)]
pub struct CanonicalHasher {
    state: Sha256,
    /// While a model is being hashed: the digest of each hoisted subtree's
    /// body, by the name the compiler gave it. `None` outside a
    /// model. See [`CanonicalHasher::with_hoisted_bodies`].
    hoisted_bodies: Option<HoistedBodies>,
}

/// The digest of each hoisted subtree's body, keyed by its compiler-assigned
/// name (`__licm_<n>`). The names are numbering, not identity; the digests are
/// what a reference to one is hashed as.
pub type HoistedBodies = std::collections::BTreeMap<String, ContentHash>;

impl Default for CanonicalHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl CanonicalHasher {
    pub fn new() -> Self {
        Self { state: Sha256::new(), hoisted_bodies: None }
    }

    /// Run `f` with `bodies` as the table that resolves a reference to a
    /// hoisted subtree, then restore whatever table was in force before. The
    /// one caller is the model's own `hash_into`: a reference means something
    /// only inside the model that holds the subtree.
    pub fn with_hoisted_bodies(&mut self, bodies: HoistedBodies, f: impl FnOnce(&mut Self)) {
        let before = self.hoisted_bodies.replace(bodies);
        f(self);
        self.hoisted_bodies = before;
    }

    /// The digest of the hoisted subtree `name`, when a model is being hashed
    /// and it holds one by that name.
    pub fn hoisted_body(&self, name: &str) -> Option<ContentHash> {
        self.hoisted_bodies.as_ref().and_then(|m| m.get(name)).copied()
    }

    /// The digest of `value` hashed on its own, resolving references to
    /// hoisted subtrees through `bodies`. Used to digest one hoisted body in
    /// terms of the bodies before it.
    pub fn digest_with_hoisted_bodies<T: ContentAddressed + ?Sized>(
        value: &T,
        bodies: &HoistedBodies,
    ) -> ContentHash {
        // A hoisted body is an expression; it holds no gradient map.
        let mut h = Self { state: Sha256::new(), hoisted_bodies: Some(bodies.clone()) };
        value.hash_into(&mut h);
        h.finalize()
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

    /// The per-type domain-separation tag, length-prefixed. Written first
    /// by every named-type `hash_into`.
    pub fn write_type_tag(&mut self, tag: &str) {
        self.write_str(tag);
    }

    /// The per-type schema version, written right after the type tag.
    pub fn write_schema_version(&mut self, v: u16) {
        self.write_u16(v);
    }

    /// Raw IEEE-754 bits of an `f64`, little-endian — the *structural IR
    /// float* policy. Distinguishes `+0.0`/`-0.0` and NaN payloads, matching
    /// the IR's own `ConstExpr::PartialEq` (`expr.rs`).
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
    /// collection order.
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

/// A type whose value has a stable structural hash: the IR types, by the
/// hand-written impls in `impls`, and the primitives below. `content_hash` is
/// the root entry point: it folds in [`HASH_VERSION`] once and delegates to
/// `hash_into`; nested fields compose through `hash_into` directly.
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
// absent: a float is written through `write_f64_bits` where it is hashed.

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
// prefix (the width is invariant). A reference to a hoisted subtree is
// hashed as its body's digest this way.
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
