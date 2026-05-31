//! Macro-equivalence golden: `#[derive(RunInput)]` output must byte-match a
//! hand-written `ContentAddressed` impl on a fixed value, before the macro is
//! trusted to replace the hand impls.
//!
//! The hand replicas live in *this* module so `module_path!()` resolves to
//! the same string the macro's type tag expands to (the macro uses
//! `concat!(module_path!(), "::", stringify!(Ident))`). They reproduce the
//! canonical framing by hand: type tag, schema version, then fields /
//! variant-index-then-payload, skipping provenance fields.

#![cfg(test)]

use crate::float::FiniteF64;
use crate::hash::{CanonicalHasher, ContentAddressed, ContentHash, HASH_VERSION};
use runid_derive::RunInput;

/// Drive a value's `hash_into` under the same root framing `content_hash`
/// uses (HASH_VERSION first), via a hand closure.
fn hand_root(f: impl FnOnce(&mut CanonicalHasher)) -> ContentHash {
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    f(&mut h);
    h.finalize()
}

// ── Struct: named fields, a FiniteF64, a Vec, and a skipped provenance field ─

#[derive(RunInput)]
struct SampleStruct {
    a: u64,
    b: FiniteF64,
    c: Vec<String>,
    #[run_input(provenance)]
    ignored: u64,
}

fn hand_struct(s: &SampleStruct, h: &mut CanonicalHasher) {
    h.write_type_tag(concat!(module_path!(), "::", "SampleStruct"));
    h.write_schema_version(1);
    s.a.hash_into(h);
    s.b.hash_into(h);
    s.c.hash_into(h);
    // `ignored` carries #[run_input(provenance)] → skipped.
}

#[test]
fn struct_macro_equals_hand() {
    let s = SampleStruct {
        a: 7,
        b: FiniteF64::new(1.5).unwrap(),
        c: vec!["x".into(), "y".into()],
        ignored: 999,
    };
    assert_eq!(s.content_hash(), hand_root(|h| hand_struct(&s, h)));
}

#[test]
fn struct_provenance_field_is_skipped() {
    let base = SampleStruct {
        a: 7,
        b: FiniteF64::new(1.5).unwrap(),
        c: vec!["x".into()],
        ignored: 1,
    };
    let other = SampleStruct {
        a: 7,
        b: FiniteF64::new(1.5).unwrap(),
        c: vec!["x".into()],
        ignored: 424_242,
    };
    // The two genuinely differ in the provenance field…
    assert_ne!(base.ignored, other.ignored);
    // …yet hash identically, because the macro skips it.
    assert_eq!(
        base.content_hash(),
        other.content_hash(),
        "changing a #[run_input(provenance)] field must not change the hash"
    );
}

// ── Enum: unit, tuple, and struct variants ──────────────────────────────────

#[derive(RunInput)]
enum SampleEnum {
    Unit,
    Tuple(u64, FiniteF64),
    Struct { x: String, y: bool },
}

fn hand_enum(e: &SampleEnum, h: &mut CanonicalHasher) {
    h.write_type_tag(concat!(module_path!(), "::", "SampleEnum"));
    h.write_schema_version(1);
    match e {
        SampleEnum::Unit => {
            h.write_u32(0);
        }
        SampleEnum::Tuple(a, b) => {
            h.write_u32(1);
            a.hash_into(h);
            b.hash_into(h);
        }
        SampleEnum::Struct { x, y } => {
            h.write_u32(2);
            x.hash_into(h);
            y.hash_into(h);
        }
    }
}

#[test]
fn enum_macro_equals_hand() {
    let values = [
        SampleEnum::Unit,
        SampleEnum::Tuple(3, FiniteF64::new(-2.0).unwrap()),
        SampleEnum::Struct { x: "hi".into(), y: true },
    ];
    for v in &values {
        assert_eq!(v.content_hash(), hand_root(|h| hand_enum(v, h)));
    }
    // Variant index discriminates: Unit ≠ Tuple ≠ Struct.
    assert_ne!(values[0].content_hash(), values[1].content_hash());
    assert_ne!(values[1].content_hash(), values[2].content_hash());
}

// ── Container schema_version override ────────────────────────────────────────

#[derive(RunInput)]
#[run_input(schema_version = 5)]
struct Versioned {
    v: u64,
}

fn hand_versioned(s: &Versioned, h: &mut CanonicalHasher) {
    h.write_type_tag(concat!(module_path!(), "::", "Versioned"));
    h.write_schema_version(5);
    s.v.hash_into(h);
}

#[test]
fn schema_version_override_matches_hand() {
    let s = Versioned { v: 1 };
    assert_eq!(s.content_hash(), hand_root(|h| hand_versioned(&s, h)));

    // And the override actually participates: a hand impl writing the default
    // version 1 must differ.
    let with_v1 = hand_root(|h| {
        h.write_type_tag(concat!(module_path!(), "::", "Versioned"));
        h.write_schema_version(1);
        s.v.hash_into(h);
    });
    assert_ne!(s.content_hash(), with_v1, "schema_version must be folded into the hash");
}
