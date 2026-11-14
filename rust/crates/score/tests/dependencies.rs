//! `score` depends on `numerics`, `serde` and `serde_json`, and on nothing
//! else: a dependency is the one way ambient access (files, processes,
//! threads, the clock, a random number generator) could enter the crate
//! without its own source naming it, which `tests/source.rs` checks.
//!
//! The manifest is read as text, with no TOML parser, so that this test adds
//! no dependency of its own. It reads the `[dependencies]` table and refuses
//! any other dependency table (`[dev-dependencies]`, `[build-dependencies]`,
//! `[target.….dependencies]`): each would be a dependency the list below does
//! not name.

use std::collections::BTreeSet;
use std::path::Path;

const ALLOWED: [&str; 3] = ["numerics", "serde", "serde_json"];

#[test]
fn the_dependencies_are_exactly_numerics_serde_and_serde_json() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let manifest = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let mut section = String::new();
    let mut found = BTreeSet::new();
    for line in manifest.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            section = line.to_string();
            assert!(
                section == "[dependencies]" || !section.contains("dependencies"),
                "score/Cargo.toml has a dependency table other than [dependencies]: {section}"
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
    assert_eq!(found, allowed, "score's [dependencies] must be exactly {ALLOWED:?}");
}
