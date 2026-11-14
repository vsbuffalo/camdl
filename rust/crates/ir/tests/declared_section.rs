//! The `declared` section of the IR envelope, from the runtime reader's side.
//!
//! Source: `ocaml/golden/sir_basic.ir.json`, a compiled golden (primary).
//!
//! The section sits beside `docs`, outside `Model`: the envelope reader keeps
//! it, the model reader drops it, and an envelope without it reads as "not
//! recorded". The runtime's reader refuses an envelope of another IR version;
//! the section itself stays readable from that same file through the
//! `declared` crate (`rust/crates/declared/tests/golden_sections.rs`).

use std::path::Path;

use serde_json::Value;

fn golden_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden/sir_basic.ir.json");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The golden with its `declared` member cut out of the text. The member is
/// the last key the compiler writes. The cut is textual on purpose: parsing to
/// a generic JSON value and writing it back would reorder object keys, and the
/// order of `initial_conditions` is part of the model.
fn golden_text_without_declared() -> String {
    let text = golden_text();
    let at = text.rfind(",\"declared\":{").expect("the golden has a trailing declared member");
    let cut = format!("{}}}", &text[..at]);
    let v: Value = serde_json::from_str(&cut).expect("the cut text is JSON");
    assert!(v.get("declared").is_none() && v.get("model").is_some());
    cut
}

#[test]
fn the_envelope_reader_keeps_the_declared_section() {
    let text = golden_text();
    let env = ir::envelope_from_str(&text).expect("envelope parses");
    let declared = env.declared.as_ref().expect("a compiled golden records the declared model");
    assert_eq!(declared.schema_version, declared::SCHEMA_VERSION);

    // What the envelope type writes back is what the compiler wrote.
    let written: Value = serde_json::to_value(&env).expect("serialize");
    let original: Value = serde_json::from_str(&text).expect("golden is JSON");
    assert_eq!(written["declared"], original["declared"]);
}

#[test]
fn an_envelope_without_the_section_reads_as_not_recorded() {
    let env = ir::envelope_from_str(&golden_text_without_declared()).expect("envelope parses");
    assert!(env.declared.is_none());
    // And writing it back does not invent one.
    let written: Value = serde_json::to_value(&env).expect("serialize");
    assert!(written.get("declared").is_none());
}

#[test]
fn the_section_is_outside_the_model() {
    let text = golden_text();
    let with = ir::from_str(&text).expect("model parses");
    let without = ir::from_str(&golden_text_without_declared()).expect("model parses");
    assert_eq!(with, without, "the model is the same with and without the declared section");
}

#[test]
fn the_runtime_reader_refuses_another_ir_version() {
    let current = format!("\"ir_version\":\"{}\"", ir::IR_VERSION.trim());
    let text = golden_text();
    assert!(text.contains(&current), "the golden is at the current IR version");
    let other = text.replacen(&current, "\"ir_version\":\"0.1\"", 1);
    let err = ir::from_str(&other).expect_err("another IR version is refused");
    assert!(matches!(err, ir::IrError::VersionMismatch { .. }), "unexpected error: {err}");
}
