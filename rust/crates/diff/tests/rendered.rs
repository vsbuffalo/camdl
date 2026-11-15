//! The text report of each fixture variant against the reference model, as
//! committed files a person can read.
//!
//! Which models: `tests/fixtures/diff_pairs` (see `tests/pairs.rs`). For each
//! variant, `expected/<variant>.txt` beside the sources is what
//! `diff::render::table` prints for `base` against it. `tests/pairs.rs` states
//! the rows; this pins how they read.
//!
//! `make update-diff-pairs-golden` rewrites the files (it runs this test with
//! `CAMDL_UPDATE_DIFF_EXPECTED=1`); a changed file is then reviewed like any
//! golden.

mod common;

use common::{compare, pair_file, repo_root};

const VARIANTS: [&str; 14] = [
    "reformatted",
    "let_renamed",
    "constant_let_renamed",
    "let_body_changed",
    "let_inlined",
    "rate_changed",
    "parameter_added",
    "transitions_reordered",
    "prior_changed",
    "observation_family_changed",
    "scenario_added",
    "simulation_end_changed",
    "quantity_changed",
    "base.set_beta",
];

#[test]
fn the_text_report_of_each_variant_is_the_committed_one() {
    let update = std::env::var_os("CAMDL_UPDATE_DIFF_EXPECTED").is_some();
    let dir = repo_root().join("tests/fixtures/diff_pairs/expected");
    let base = pair_file("base");
    let mut stale = Vec::new();
    for variant in VARIANTS {
        let report = compare(&base, &pair_file(variant)).unwrap_or_else(|e| panic!("base against {variant}: {e}"));
        let text = diff::render::table(&report);
        let path = dir.join(format!("{variant}.txt"));
        if update {
            std::fs::create_dir_all(&dir).expect("create the expected directory");
            std::fs::write(&path, &text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        } else if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
            stale.push(variant);
        }
    }
    // The stratified pair has its own reference model.
    let report = compare(&pair_file("age_base"), &pair_file("age_level_added")).expect("a report");
    let text = diff::render::table(&report);
    let path = dir.join("age_level_added.txt");
    if update {
        std::fs::write(&path, &text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    } else if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
        stale.push("age_level_added");
    }
    assert!(
        stale.is_empty(),
        "the text report changed for {stale:?}; run `make update-diff-pairs-golden` and review the files"
    );
}
