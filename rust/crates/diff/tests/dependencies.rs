//! `diff` depends on `declared`, `serde` and `serde_json`, and on nothing
//! else. In particular not on `ir`, `runid`, `sim` or `cli`: the diff of two
//! published declared-model documents must not need the reader for compiled
//! models, and the function must stay callable from a viewer.
//!
//! The manifest is read as text. `[dev-dependencies]` is allowed and not
//! checked: the tests compute model hashes with `runid`, which the library
//! never does.

use std::collections::BTreeSet;
use std::path::Path;

const ALLOWED: [&str; 3] = ["declared", "serde", "serde_json"];

#[test]
fn the_dependencies_are_exactly_declared_serde_and_serde_json() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut section = String::new();
    let mut found = BTreeSet::new();
    for line in manifest.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            section = line.to_string();
            assert!(
                section == "[dependencies]" || section == "[dev-dependencies]" || !section.contains("dependencies"),
                "diff/Cargo.toml has an unexpected dependency table: {section}"
            );
            continue;
        }
        if section == "[dependencies]" {
            // `name = …` or `name.workspace = true`.
            let name = line.split(['=', '.']).next().unwrap_or("").trim();
            found.insert(name.to_string());
        }
    }

    let allowed: BTreeSet<String> = ALLOWED.iter().map(|s| s.to_string()).collect();
    assert_eq!(found, allowed, "diff's [dependencies] must be exactly {ALLOWED:?}");
}
