//! Property tests for the canonical hashing core. These pin the encoding
//! invariants that make content-addressing sound: equal values hash equal,
//! and structurally-distinct values do not collide.

use std::collections::HashMap;

use crate::float::FiniteF64;
use crate::hash::{CanonicalHasher, ContentAddressed, ContentHash};
use crate::kind::{run_id, ArtifactKind};

/// Hash a single value through a fresh hasher (no `HASH_VERSION` framing) —
/// a test helper for exercising encoding rules directly.
fn raw_hash(f: impl FnOnce(&mut CanonicalHasher)) -> ContentHash {
    let mut h = CanonicalHasher::new();
    f(&mut h);
    h.finalize()
}

// ── Length-prefixing kills concatenation ambiguity ──────────────────────────

#[test]
fn length_prefix_separates_strings() {
    // ("ab","c") and ("a","bc") concatenate to the same bytes; the length
    // prefix must keep them distinct.
    let a = raw_hash(|h| {
        h.write_str("ab");
        h.write_str("c");
    });
    let b = raw_hash(|h| {
        h.write_str("a");
        h.write_str("bc");
    });
    assert_ne!(a, b, "length-prefixing must separate (\"ab\",\"c\") from (\"a\",\"bc\")");
}

#[test]
fn vec_is_count_prefixed_and_order_sensitive() {
    let ab = raw_hash(|h| vec!["a".to_string(), "b".to_string()].hash_into(h));
    let ba = raw_hash(|h| vec!["b".to_string(), "a".to_string()].hash_into(h));
    let single = raw_hash(|h| vec!["ab".to_string()].hash_into(h));
    assert_ne!(ab, ba, "Vec order is significant");
    assert_ne!(ab, single, "count prefix separates [a,b] from [ab]");
}

#[test]
fn option_tag_distinguishes_none_some_and_nesting() {
    let none = raw_hash(|h| Option::<u64>::None.hash_into(h));
    let some0 = raw_hash(|h| Some(0u64).hash_into(h));
    // None is the 0 byte; Some(0) is 1 followed by eight zero bytes — the
    // tag byte must keep them apart.
    assert_ne!(none, some0);
}

// ── Maps hash in sorted key order, regardless of iteration order ─────────────

#[test]
fn str_f64_map_is_order_invariant() {
    // Permuting insertion order of a HashMap must not change the hash — the
    // sorted-iteration rule. Build two maps with different insertion orders.
    let mut m1: HashMap<String, f64> = HashMap::new();
    m1.insert("beta".into(), 2.0);
    m1.insert("alpha".into(), 1.0);
    m1.insert("gamma".into(), 3.0);

    let mut m2: HashMap<String, f64> = HashMap::new();
    m2.insert("gamma".into(), 3.0);
    m2.insert("alpha".into(), 1.0);
    m2.insert("beta".into(), 2.0);

    let h1 = raw_hash(|h| h.write_str_f64_map(m1.iter()));
    let h2 = raw_hash(|h| h.write_str_f64_map(m2.iter()));
    assert_eq!(h1, h2, "map hash must be invariant under insertion order");

    // Negative control: a different value must change the hash.
    let mut m3 = m1.clone();
    m3.insert("beta".into(), 2.5);
    let h3 = raw_hash(|h| h.write_str_f64_map(m3.iter()));
    assert_ne!(h1, h3, "changing a map value must change the hash");
}

#[test]
fn str_map_of_content_addressed_is_order_invariant() {
    let mut m1: HashMap<String, u64> = HashMap::new();
    m1.insert("b".into(), 20);
    m1.insert("a".into(), 10);
    let mut m2: HashMap<String, u64> = HashMap::new();
    m2.insert("a".into(), 10);
    m2.insert("b".into(), 20);
    let h1 = raw_hash(|h| h.write_str_map(m1.iter()));
    let h2 = raw_hash(|h| h.write_str_map(m2.iter()));
    assert_eq!(h1, h2);
}

// ── Float policies ───────────────────────────────────────────────────────────

#[test]
fn finite_f64_normalizes_negative_zero() {
    let pos = FiniteF64::new(0.0).unwrap();
    let neg = FiniteF64::new(-0.0).unwrap();
    assert_eq!(
        pos.content_hash(),
        neg.content_hash(),
        "resolved-input policy must hash -0.0 and +0.0 identically"
    );
    // Negative control: a genuinely different value differs.
    let one = FiniteF64::new(1.0).unwrap();
    assert_ne!(pos.content_hash(), one.content_hash());
}

#[test]
fn finite_f64_rejects_non_finite() {
    assert!(FiniteF64::new(f64::NAN).is_err(), "NaN must be rejected before hashing");
    assert!(FiniteF64::new(f64::INFINITY).is_err(), "+Inf must be rejected");
    assert!(FiniteF64::new(f64::NEG_INFINITY).is_err(), "-Inf must be rejected");
    // A finite value at the extreme is fine.
    assert!(FiniteF64::new(f64::MAX).is_ok());
}

#[test]
fn structural_float_bits_distinguish_signed_zero() {
    // The structural IR-float policy keeps ±0.0 distinct (matching
    // ConstExpr::PartialEq, which compares to_bits()).
    let pos = raw_hash(|h| h.write_f64_bits(0.0));
    let neg = raw_hash(|h| h.write_f64_bits(-0.0));
    assert_ne!(pos, neg, "structural floats must distinguish +0.0 from -0.0");
}

// ── run_id root derivation ───────────────────────────────────────────────────

fn fake_hash(byte: u8) -> ContentHash {
    ContentHash::from_bytes([byte; 32])
}

#[test]
fn run_id_is_deterministic() {
    let levels = [fake_hash(1), fake_hash(2)];
    assert_eq!(run_id(ArtifactKind::Sim, &levels), run_id(ArtifactKind::Sim, &levels));
}

#[test]
fn run_id_kind_tag_prevents_aliasing() {
    // Two kinds with an identical level sequence must not produce the same
    // run_id — the fixed-width kind tag separates them.
    let levels = [fake_hash(1), fake_hash(2)];
    assert_ne!(
        run_id(ArtifactKind::Sim, &levels),
        run_id(ArtifactKind::FitStage, &levels),
        "kind tag must disambiguate equal level sequences"
    );
}

#[test]
fn run_id_count_prefix_prevents_level_concatenation_collision() {
    // [h_ab] must not collide with [h_a, h_b] for any decomposition — the
    // count prefix frames the list. Here we use single-byte-distinct hashes:
    // a 2-level list and a 1-level list can never share bytes given the
    // length prefix.
    let two = [fake_hash(1), fake_hash(2)];
    let one = [fake_hash(1)];
    assert_ne!(run_id(ArtifactKind::Sim, &two), run_id(ArtifactKind::Sim, &one));
}

// ── ContentHash hex round-trip ───────────────────────────────────────────────

#[test]
fn content_hash_hex_roundtrip() {
    let h = raw_hash(|hh| hh.write_str("anything"));
    let hex = h.to_hex();
    assert_eq!(hex.len(), 64, "full hex is 64 chars");
    assert_eq!(h.short8().len(), 8, "short form is 8 hex chars");
    assert_eq!(h.short8(), &hex[..8]);
    assert_eq!(ContentHash::from_hex(&hex).unwrap(), h, "hex round-trips");
    assert!(ContentHash::from_hex("not-hex").is_err());
    assert!(ContentHash::from_hex("abcd").is_err(), "wrong length rejected");
}

#[test]
fn content_hash_serde_is_hex_string() {
    let h = fake_hash(0xab);
    let json = serde_json::to_string(&h).unwrap();
    assert_eq!(json, format!("\"{}\"", "ab".repeat(32)));
    let back: ContentHash = serde_json::from_str(&json).unwrap();
    assert_eq!(back, h);
}

// ── HASH_VERSION is folded into the root ─────────────────────────────────────

#[test]
fn content_hash_folds_hash_version() {
    // content_hash() prepends HASH_VERSION; a bare hash_into of the same
    // value (no version) must therefore differ from content_hash().
    let v = 42u64;
    let with_version = v.content_hash();
    let without = raw_hash(|h| v.hash_into(h));
    assert_ne!(with_version, without, "content_hash must fold in HASH_VERSION");
}
