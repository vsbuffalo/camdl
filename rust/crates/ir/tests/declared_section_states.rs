//! The three states of an envelope's `declared` section, and what happens to
//! the section when a compiled model is read, changed and written back.
//!
//! Source: `ocaml/golden/sir_basic.ir.json`, a compiled golden (primary).
//!
//! Every edit to the golden's text here is textual. Parsing a compiled model
//! to a generic JSON value and writing it back reorders object keys, and the
//! order of `initial_conditions` is part of the model
//! (`rust/crates/ir/tests/value_round_trip.rs`).

use std::path::Path;

use declared::{DeclaredError, DeclaredSection, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS};

fn golden_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden/sir_basic.ir.json");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// How the compiler opens the section: `schema_version` is its first member.
fn section_opening() -> String {
    format!(",\"declared\":{{\"schema_version\":{SCHEMA_VERSION}")
}

/// The golden with its section's `schema_version` changed to one this build
/// does not read, and a member added that version 1 does not have.
fn golden_text_with_future_section() -> String {
    let text = golden_text();
    let opening = section_opening();
    assert!(text.contains(&opening), "the golden's section opens with its schema_version");
    text.replacen(&opening, ",\"declared\":{\"schema_version\":999,\"added_in_999\":[1,2]", 1)
}

/// The text of the `declared` member of an envelope, exactly as written.
fn declared_member_text(envelope: &str) -> String {
    #[derive(serde::Deserialize)]
    struct OnlyDeclared {
        declared: Box<serde_json::value::RawValue>,
    }
    serde_json::from_str::<OnlyDeclared>(envelope)
        .expect("an envelope with a declared member")
        .declared
        .get()
        .to_string()
}

#[test]
fn a_section_of_an_unknown_version_is_a_typed_state_and_the_model_still_loads() {
    let text = golden_text_with_future_section();

    // The model loads: the section is not the model.
    let model = ir::from_str(&text).expect("the model loads beside an unreadable section");
    assert_eq!(model, ir::from_str(&golden_text()).expect("model parses"));

    // The section is neither absent nor an error.
    let env = ir::envelope_from_str(&text).expect("the envelope loads");
    match &env.declared {
        DeclaredSection::UnsupportedVersion { schema_version, .. } => assert_eq!(*schema_version, 999),
        other => panic!("expected an unsupported-version section, got {other:?}"),
    }
    assert!(!env.declared.is_not_recorded());

    // Asking for the declared model names the version found and those read.
    let err = env.declared.recorded().expect_err("an unsupported section has no readable model");
    assert_eq!(err, DeclaredError::UnsupportedVersion { found: 999, supported: SUPPORTED_SCHEMA_VERSIONS });
    assert_eq!(err.to_string(), "declared model has schema_version 999; this build reads 1");
}

#[test]
fn an_unsupported_section_is_written_back_as_the_text_it_was_read_from() {
    let text = golden_text_with_future_section();
    let env = ir::envelope_from_str(&text).expect("the envelope loads");
    let written = serde_json::to_string(&env).expect("serialize");
    assert_eq!(declared_member_text(&written), declared_member_text(&text));

    // Through a rewrite of the model as well.
    let mut model = env.model.clone();
    model.simulation.t_end += 1.0;
    let rewritten = ir::reemit_with_model(&text, &model).expect("re-emit");
    let back = ir::envelope_from_str(&rewritten).expect("the rewritten envelope parses");
    assert_eq!(back.declared, env.declared);
}

#[test]
fn a_malformed_section_of_a_known_version_is_refused() {
    // Version 1 with a member version 1 does not have is not a newer document
    // but a broken one. The envelope is refused, and not read with a section
    // that silently lost something.
    let text = golden_text();
    let opening = section_opening();
    let broken = text.replacen(&opening, &format!("{opening},\"not_a_member\":true"), 1);
    let err = ir::envelope_from_str(&broken).expect_err("a malformed section is refused");
    assert!(err.to_string().contains("not_a_member"), "{err}");
}

#[test]
fn an_explicit_null_section_reads_as_not_recorded() {
    let text = golden_text();
    let at = text.rfind(",\"declared\":{").expect("the golden has a trailing declared member");
    let nulled = format!("{},\"declared\":null}}", &text[..at]);
    let env = ir::envelope_from_str(&nulled).expect("envelope parses");
    assert!(env.declared.is_not_recorded());
}

#[test]
fn rewriting_a_compiled_model_carries_docs_and_the_declared_section_through() {
    let text = golden_text();
    let original = ir::envelope_from_str(&text).expect("envelope parses");
    assert!(original.declared.recorded().is_ok());

    // What a command does: read, change the model, write back.
    let mut model = original.model.clone();
    model.simulation.t_end += 1.0;
    let rewritten = ir::reemit_with_model(&text, &model).expect("re-emit");
    let back = ir::envelope_from_str(&rewritten).expect("the rewritten envelope parses");

    assert_eq!(back.model, model, "the changed model is what was written");
    assert_eq!(back.declared, original.declared, "the declared section is unchanged");
    assert_eq!(
        serde_json::to_string(&back.docs).expect("serialize docs"),
        serde_json::to_string(&original.docs).expect("serialize docs"),
        "the doc dictionary is unchanged"
    );
    assert!(back.validated_by.is_none(), "the compiler did not validate the rewritten model");
}

#[test]
fn wrapping_a_hand_built_model_states_that_no_section_is_recorded() {
    let model = ir::from_str(&golden_text()).expect("model parses");
    let env = ir::IrEnvelope::wrap(model, None, DeclaredSection::NotRecorded);
    let written = serde_json::to_string(&env).expect("serialize");
    assert!(!written.contains("\"declared\""));
}
