//! Invariant 9 of `docs/dev/proposals/2026-11-13-model-structure-diff.md`: the
//! `declared` section of the IR envelope sits outside the model, so adding it
//! changes no model identity.
//!
//! `fixtures/golden_model_hashes.tsv` holds `model_ir_hash` for every compiled
//! golden model (every `*.ir.json` that `make update-golden` regenerates),
//! recorded at the commit named in its header, which is the last one before
//! the section existed. This test recomputes each hash from the golden files
//! as they are now and compares. A golden whose identity moved is a model that
//! was re-keyed; that is never collateral.
//!
//! The table is regenerated only by `make update-golden-model-hashes`, and
//! only when a re-key is the subject of the change.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const TABLE: &str = "tests/fixtures/golden_model_hashes.tsv";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Every compiled golden model, as a path relative to the repository root:
/// `ocaml/golden/*.ir.json` and `tests/fixtures/*/ir/*.ir.json`.
fn golden_files() -> Vec<String> {
    let root = repo_root();
    let mut dirs = vec![root.join("ocaml/golden")];
    let fixtures = root.join("tests/fixtures");
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixtures.display()))
        .map(|entry| entry.expect("directory entry").path().join("ir"))
        .filter(|ir| ir.is_dir())
        .collect();
    subdirs.sort();
    dirs.extend(subdirs);

    let mut files = Vec::new();
    for dir in dirs {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
            let path = entry.expect("directory entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string();
            if name.ends_with(".ir.json") {
                let rel = path.strip_prefix(&root).expect("under the repository root");
                // The root is `<manifest>/../../..`, so the relative part is
                // already normalized below it.
                files.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    files.sort();
    files
}

fn model_hash(rel: &str) -> String {
    let path = repo_root().join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let model = ir::from_str(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
    ir::model_hash::model_ir_hash(&model).to_hex()
}

fn read_table() -> (Vec<String>, BTreeMap<String, String>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(TABLE);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let header: Vec<String> = text.lines().filter(|l| l.starts_with('#')).map(str::to_string).collect();
    let rows = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .skip(1) // column names
        .map(|l| {
            let (file, hash) = l.split_once('\t').unwrap_or_else(|| panic!("malformed row: {l}"));
            (file.to_string(), hash.to_string())
        })
        .collect();
    (header, rows)
}

#[test]
fn every_golden_model_keeps_its_recorded_identity() {
    let files = golden_files();

    // `make update-golden-model-hashes` sets this, naming the commit the
    // goldens are being recorded at.
    if let Ok(commit) = std::env::var("CAMDL_CAPTURE_GOLDEN_MODEL_HASHES") {
        let mut out = String::new();
        out.push_str("# model_ir_hash of every compiled golden model.\n");
        out.push_str("# Source: the committed *.ir.json files under ocaml/golden and tests/fixtures/*/ir (primary).\n");
        out.push_str(&format!("# Recorded at commit: {commit}\n"));
        out.push_str("# Regenerate: make update-golden-model-hashes\n");
        out.push_str("file\tmodel_ir_hash\n");
        for f in &files {
            out.push_str(&format!("{f}\t{}\n", model_hash(f)));
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(TABLE);
        std::fs::write(&path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }

    // `make add-golden-model-hashes` sets this, naming the commit at which new
    // golden files are being added. Existing rows and the recorded-at line are
    // kept exactly; only files with no row get one. A golden whose identity
    // moved is not helped by this mode, and still fails the comparison below
    // on the next plain run.
    if let Ok(commit) = std::env::var("CAMDL_ADD_GOLDEN_MODEL_HASHES") {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(TABLE);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let (_, recorded) = read_table();
        let new: Vec<&String> = files.iter().filter(|f| !recorded.contains_key(*f)).collect();
        if new.is_empty() {
            return;
        }
        let mut header: Vec<String> = Vec::new();
        let mut rows: Vec<String> = Vec::new();
        let mut columns = String::new();
        for line in text.lines() {
            if line.starts_with('#') {
                header.push(line.to_string());
            } else if columns.is_empty() {
                columns = line.to_string();
            } else if !line.trim().is_empty() {
                rows.push(line.to_string());
            }
        }
        let names: Vec<&str> = new.iter().map(|f| f.as_str()).collect();
        header.push(format!("# Rows added at commit {commit}: {}", names.join(", ")));
        for f in &new {
            rows.push(format!("{f}\t{}", model_hash(f)));
        }
        // Rows are keyed by file and kept sorted, as the full capture writes them.
        rows.sort();
        let mut out = header.join("\n");
        out.push('\n');
        out.push_str(&columns);
        out.push('\n');
        out.push_str(&rows.join("\n"));
        out.push('\n');
        std::fs::write(&path, out).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }

    let (_, recorded) = read_table();
    let mut problems = Vec::new();
    for f in &files {
        match recorded.get(f) {
            None => problems.push(format!("{f}: a golden with no recorded identity")),
            Some(want) => {
                let got = model_hash(f);
                if &got != want {
                    problems.push(format!("{f}: recorded {want}, now {got}"));
                }
            }
        }
    }
    for f in recorded.keys() {
        if !files.contains(f) {
            problems.push(format!("{f}: recorded, but the golden no longer exists"));
        }
    }
    assert!(
        problems.is_empty(),
        "golden model identities moved ({} of {}):\n  {}\n\
         If a re-key is the subject of this change, run `make update-golden-model-hashes`.",
        problems.len(),
        files.len(),
        problems.join("\n  ")
    );
    assert!(files.len() >= 90, "expected the whole golden set, found {} files", files.len());
}
