//! The display-only members sit only where the diff assumes they do.
//!
//! The diff leaves `diff::document::DISPLAY_ONLY` out of every comparison, by
//! member name. That is sound only if those names are never used for anything
//! else. This test reads the `declared` section of every compiled golden as
//! raw JSON and checks where each name occurs:
//!
//! - `katex`, `katex_loss` and `reads` only as members of an expression, the
//!   object that also has `tree`;
//! - `source_text` and `expands_to` only as members of a declaration: an entry
//!   of one of the section's lists, or its `simulation` or `output` member.
//!
//! Source: the committed compiled models (primary: what the compiler emits).

mod common;

use common::golden_files;
use diff::document::DISPLAY_ONLY;
use serde_json::Value;

const OF_AN_EXPRESSION: [&str; 3] = ["katex", "katex_loss", "reads"];
const OF_A_DECLARATION: [&str; 2] = ["source_text", "expands_to"];

/// `is_declaration`: the object is an entry of a top-level list, or the
/// `simulation` or `output` member.
fn check(file: &str, path: &str, value: &Value, is_declaration: bool, seen: &mut [usize; 5]) {
    match value {
        Value::Object(members) => {
            for (key, child) in members {
                if let Some(i) = DISPLAY_ONLY.iter().position(|d| d == key) {
                    seen[i] += 1;
                    if OF_AN_EXPRESSION.contains(&key.as_str()) {
                        assert!(members.contains_key("tree"), "{file}: {path}.{key} is not on an expression");
                    } else {
                        assert!(is_declaration, "{file}: {path}.{key} is not on a declaration");
                    }
                    if key != "reads" {
                        // `reads` holds names; nothing under the others can
                        // hold a display-only member of its own.
                        continue;
                    }
                }
                check(file, &format!("{path}.{key}"), child, false, seen);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                check(file, &format!("{path}[{i}]"), item, false, seen);
            }
        }
        _ => {}
    }
}

#[test]
fn display_only_members_occur_only_on_expressions_and_declarations() {
    assert_eq!(
        DISPLAY_ONLY.len(),
        OF_AN_EXPRESSION.len() + OF_A_DECLARATION.len(),
        "a display-only member was added without saying where it may occur"
    );
    let mut seen = [0usize; 5];
    let files = golden_files();
    for path in &files {
        let file = path.file_name().expect("a file name").to_string_lossy().into_owned();
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let envelope: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{file} is not JSON: {e}"));
        let declared = envelope.get("declared").and_then(Value::as_object).unwrap_or_else(|| panic!("{file}: no declared section"));
        for (member, value) in declared {
            match value {
                Value::Array(entries) => {
                    for (i, entry) in entries.iter().enumerate() {
                        check(&file, &format!("{member}[{i}]"), entry, true, &mut seen);
                    }
                }
                other => check(&file, member, other, member == "simulation" || member == "output", &mut seen),
            }
        }
    }
    // The test bites: every member was met.
    for (name, count) in DISPLAY_ONLY.iter().zip(seen) {
        assert!(count > 0, "no golden has a `{name}` member; the check above saw nothing");
    }
}
