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

/// Every kind camdl registers on the store has a file contract, so an output
/// manifest can name the files of any result.
#[test]
fn every_registered_kind_has_a_contract() {
    for kind in crate::camdl_store::camdl_kinds().names() {
        assert!(contract(kind).is_some(), "`{kind}` has no file contract");
    }
}

/// Each file the contract lists is read back from its stored name to its
/// logical name: what the output manifest calls each stored file.
#[test]
fn every_contract_file_is_named_from_its_stored_name() {
    for (kind, _, rows) in contract::kinds() {
        let c = contract(kind).unwrap_or_else(|| panic!("`{kind}` has no contract"));
        for r in rows {
            let stored = r.stored.replace("chain_N", "chain_3");
            let logical = r.logical.replace("chain_N", "chain_3");
            assert_eq!(c.logical(&stored), Some(logical), "{kind}: `{stored}`");
        }
    }
}

#[test]
fn a_content_artifact_names_its_one_file_by_its_stem() {
    assert_eq!(contract("compiled_model").unwrap().logical("model.ir.json").as_deref(), Some("model"));
    assert_eq!(contract("quantities").unwrap().logical("quantities.json").as_deref(), Some("quantities"));
    assert_eq!(contract("data_table").unwrap().logical("table.tsv").as_deref(), Some("table"));
    assert_eq!(contract("data_table").unwrap().logical("other.tsv"), None);
}

/// Each kind's contract version, pinned to a digest of the files its
/// contract lists: each file's logical name, stored name, format and
/// columns, in the contract's order. A change to a file enum that adds,
/// removes or renames a file, or changes a column, changes the digest and
/// fails this test until the kind's `CONTRACT` is bumped and its pin
/// updated, so a workflow reading the manifest's `contract` sees the change.
#[test]
fn every_contract_version_is_pinned_to_its_files() {
    use sha2::Digest;
    const PINS: &[(&str, u16, &str)] = &[
        ("fit_run", 1, "dd2bd91bcb57e9e3"),
        ("model_source", 1, "5e161da1b9744beb"),
        ("fit", 1, "25b195434f1880a2"),
        ("prediction", 2, "99cfdc3334a62eeb"),
        ("score", 2, "1ddbae198ee2d516"),
        ("pfilter", 1, "c390dcc138c34252"),
        ("trace", 1, "55473bda7e3e6ee3"),
        ("pf_health", 1, "134c84fddede4fc6"),
        ("filtering", 1, "6feabcf765fa6960"),
        ("filtered_streams", 1, "bfa5007afd1eca58"),
        ("final_state", 1, "540e45de2abcd932"),
        ("saved_paths", 2, "db7b52705e4c11fe"),
        ("prequential", 1, "e2fbc4e24518b585"),
        ("sim", 2, "8f7955f955526563"),
        ("obs", 1, "35903996fb59dba8"),
        ("event_log", 1, "e063ab91a24119ac"),
        ("sim_quantities", 1, "54d153eb11c03a3c"),
        ("profile_point", 1, "ae106f82ec9c913d"),
        ("compiled_model", 1, "5690331bcc83e624"),
        ("quantities", 1, "319b42ae8e72d9df"),
        ("data_table", 1, "6a6fff53e18b08d4"),
    ];
    let mut digests: Vec<(&str, u16, String)> = contract::kinds()
        .into_iter()
        .map(|(kind, _, rows)| {
            let mut h = sha2::Sha256::new();
            for r in rows {
                h.update(format!("{}\t{}\t{}\t{:?}\n", r.logical, r.stored, r.format, r.columns).as_bytes());
            }
            (kind, contract(kind).unwrap().version, hex::encode(&h.finalize()[..8]))
        })
        .collect();
    use store::{Artifact, ContentArtifact};
    use crate::camdl_store as cs;
    for (kind, file) in [
        (cs::CompiledModel::KIND, cs::CompiledModel::FILE),
        (cs::Quantities::KIND, cs::Quantities::FILE),
        (cs::DataTable::KIND, cs::DataTable::FILE),
    ] {
        let logical = contract(kind).unwrap().logical(file).unwrap();
        let d = sha2::Sha256::digest(format!("{logical}\t{file}\n").as_bytes());
        digests.push((kind, contract(kind).unwrap().version, hex::encode(&d[..8])));
    }
    let pinned: Vec<(&str, u16, String)> = PINS.iter().map(|(k, v, d)| (*k, *v, d.to_string())).collect();
    assert_eq!(
        digests, pinned,
        "a kind's files changed: bump its CONTRACT and update its pin here (the left side is what the enums give now)"
    );
}

/// The sampler summary is named `sampler_summary` whichever MCMC sampler
/// wrote it, though the contract lists the MH file alone.
#[test]
fn every_samplers_summary_is_the_sampler_summary() {
    let c = contract("fit_run").unwrap();
    for stored in ["mh_summary.json", "pmmh_summary.json", "pgas_summary.json"] {
        assert_eq!(c.logical(stored).as_deref(), Some("sampler_summary"), "{stored}");
    }
}

#[test]
fn a_name_no_result_of_the_kind_stores_is_refused() {
    assert_eq!(contract("fit_run").unwrap().logical("chain_x/trace.tsv"), None);
    assert_eq!(contract("fit_run").unwrap().logical("nonsense.tsv"), None);
    assert_eq!(contract("sim").unwrap().logical("obs/cases.tsv"), None);
    assert_eq!(contract("obs").unwrap().logical("nested/cases.tsv"), None);
    assert!(contract("no_such_kind").is_none());
}

/// A name a kind's contract lists, with `chain_N` and `<...>` read as any
/// chain and any one path segment, is one the kind can hold; a name it does
/// not list is not.
#[test]
fn kind_names_reads_the_contracts_templates() {
    assert!(kind_names("model_source", "render"));
    assert!(kind_names("model_source", "model.render.json"));
    assert!(!kind_names("model_source", "rendr"));
    assert!(kind_names("fit_run", "draws"));
    assert!(kind_names("fit_run", "chain_12/trace"));
    assert!(kind_names("fit_run", "chain_3/trace.tsv"));
    assert!(!kind_names("fit_run", "chain_/trace"));
    assert!(!kind_names("fit_run", "chain_x/trace"));
    assert!(kind_names("prediction", "predictive/cases"));
    assert!(!kind_names("prediction", "predictive/a/b"));
    assert!(kind_names("compiled_model", "model.ir.json"));
    assert!(!kind_names("no_such_kind", "draws"));
}
