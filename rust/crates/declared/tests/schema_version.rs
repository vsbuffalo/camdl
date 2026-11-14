//! The schema version is checked by this crate, where a document is read, so
//! a reader with no `ir` crate gets the same refusal.
//!
//! Source: the `declared` member of `ocaml/golden/sir_basic.ir.json`, a
//! compiled golden (primary), taken out as text.

use std::path::Path;

use declared::{DeclaredError, DeclaredModel, SchemaVersion, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS};

/// The text of the golden's `declared` member, exactly as the compiler wrote it.
fn golden_section_text() -> String {
    #[derive(serde::Deserialize)]
    struct OnlyDeclared {
        declared: Box<serde_json::value::RawValue>,
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden/sir_basic.ir.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str::<OnlyDeclared>(&text).expect("an envelope with a declared member").declared.get().to_string()
}

fn opening() -> String {
    format!("{{\"schema_version\":{SCHEMA_VERSION}")
}

#[test]
fn the_current_version_reads() {
    let model = DeclaredModel::from_json_str(&golden_section_text()).expect("version 1 reads");
    assert_eq!(model.schema_version, SchemaVersion::CURRENT);
    assert!(!model.compartments.is_empty());
}

#[test]
fn a_document_of_an_unknown_version_is_refused_naming_both_versions() {
    let text = golden_section_text();
    assert!(text.starts_with(&opening()), "the section opens with its schema_version");

    // A newer document, with a member version 1 does not have. The version is
    // what is reported, not the unknown member.
    let future = text.replacen(&opening(), "{\"schema_version\":7,\"added_in_7\":null", 1);
    let err = DeclaredModel::from_json_str(&future).expect_err("version 7 is not read");
    assert_eq!(err, DeclaredError::UnsupportedVersion { found: 7, supported: SUPPORTED_SCHEMA_VERSIONS });
    assert_eq!(err.to_string(), "declared model has schema_version 7; this build reads 1");
}

#[test]
fn the_type_itself_cannot_hold_an_unknown_version() {
    // Through the plain derive, with no call to `from_json_str`.
    let bare = golden_section_text().replacen(&opening(), "{\"schema_version\":7", 1);
    let err = serde_json::from_str::<DeclaredModel>(&bare).expect_err("the type refuses version 7");
    assert!(err.to_string().contains("schema_version 7"), "{err}");

    assert!(SchemaVersion::new(SCHEMA_VERSION).is_ok());
    assert!(SchemaVersion::new(0).is_err());
    assert!(SchemaVersion::new(SCHEMA_VERSION + 1).is_err());
}

#[test]
fn a_document_with_no_version_is_malformed() {
    let err = DeclaredModel::from_json_str("{\"compartments\":[]}").expect_err("no version");
    assert!(matches!(err, DeclaredError::Malformed(_)), "{err:?}");
}
