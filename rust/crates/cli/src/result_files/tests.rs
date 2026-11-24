use super::*;

const C: [Column; 6] = [
    Column::Named("chain_id"),
    Column::Named("attempt"),
    Column::Each("each estimated parameter"),
    Column::Named("ess"),
    Column::Each("anything"),
    Column::Named("reason"),
];

#[test]
fn a_header_matches_its_named_columns_around_the_runs() {
    assert!(header_matches(&C, &["chain_id", "attempt", "beta", "ess", "reason"]));
    assert!(header_matches(&C, &["chain_id", "attempt", "ess", "reason"]));
    assert!(header_matches(&C, &["chain_id", "attempt", "beta", "gamma", "ess", "x", "reason"]));
    assert!(!header_matches(&C, &["chain_id", "attempt", "beta", "reason"]), "a named column between the runs is missing");
    assert!(!header_matches(&C, &["attempt", "chain_id", "ess", "reason"]), "the opening columns are out of order");
    assert!(!header_matches(&C, &["chain_id", "attempt", "ess", "reason", "extra"]), "the closing column is not last");
}

#[test]
fn a_header_without_runs_matches_exactly() {
    let c = [Column::Named("t"), Column::Named("log_score")];
    assert!(header_matches(&c, &["t", "log_score"]));
    assert!(!header_matches(&c, &["t", "log_score", "ess"]));
    assert!(!header_matches(&c, &["t"]));
}

/// The run spec's result-file contract, between its two marker lines, is the
/// one the file enums render, up to the reflowing of whitespace. With
/// `CAMDL_UPDATE_RESULT_FILE_CONTRACT=1` (`make update-result-file-contract`)
/// the section is written instead.
#[test]
fn the_contract_section_is_generated() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/camdl-run-spec.md");
    let text = std::fs::read_to_string(&path).unwrap();
    let begin = text.find(contract::BEGIN).expect("the run spec holds the contract's opening marker");
    let end = text.find(contract::END).expect("the run spec holds the contract's closing marker");
    let body_start = begin + contract::BEGIN.len();
    let rendered = contract::render();
    if std::env::var_os("CAMDL_UPDATE_RESULT_FILE_CONTRACT").is_some() {
        let updated = format!("{}\n\n{}\n{}", &text[..body_start], rendered, &text[end..]);
        std::fs::write(&path, updated).unwrap();
        return;
    }
    assert_eq!(
        contract::normalized(&text[body_start..end]),
        contract::normalized(&rendered),
        "docs/camdl-run-spec.md §2.9 differs from the file enums; run `make update-result-file-contract`"
    );
}

/// Every kind camdl registers on the store, except the three content
/// artifacts `Store::put` stores, has a section in the contract.
#[test]
fn the_contract_names_every_claimed_kind() {
    let rendered = contract::render();
    let content = ["compiled_model", "quantities", "data_table"];
    for kind in crate::camdl_store::camdl_kinds().names() {
        if !content.contains(&kind) {
            assert!(rendered.contains(&format!("#### `{kind}`")), "the contract has no section for `{kind}`");
        }
    }
}

/// The contract's comparison ignores a reflow and nothing else: a list
/// item's text rewrapped compares equal; two items merged into one, a heading
/// joined to its paragraph and a paragraph split in two do not.
#[test]
fn the_contract_comparison_keeps_the_blocks() {
    let n = contract::normalized;
    let committed = "#### `sim`\n\nOne simulated cell.\n\n- `traj`: `traj.tsv`, TSV (when the engine ran).\n- `failure`: `failure.toml`, TOML.\n";
    let reflowed = "#### `sim`\n\nOne simulated\ncell.\n\n- `traj`: `traj.tsv`,\n  TSV (when the engine ran).\n- `failure`:   `failure.toml`, TOML.\n";
    assert_eq!(n(committed), n(reflowed));
    let merged = "#### `sim`\n\nOne simulated cell.\n\n- `traj`: `traj.tsv`, TSV (when the engine ran). `failure`: `failure.toml`, TOML.\n";
    assert_ne!(n(committed), n(merged), "two list items merged");
    let joined = "#### `sim` One simulated cell.\n\n- `traj`: `traj.tsv`, TSV (when the engine ran).\n- `failure`: `failure.toml`, TOML.\n";
    assert_ne!(n(committed), n(joined), "a heading joined to its paragraph");
    let split = "#### `sim`\n\nOne simulated\n\ncell.\n\n- `traj`: `traj.tsv`, TSV (when the engine ran).\n- `failure`: `failure.toml`, TOML.\n";
    assert_ne!(n(committed), n(split), "a paragraph split in two");
}
