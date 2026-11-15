//! Loading compiled models for the diff tests.
//!
//! Source: committed `*.ir.json` files (primary: what the compiler emits). The
//! declared section is read from the envelope and the model hash is computed by
//! `runid::inputs::model_ir_hash`, the one model identity; the diff itself
//! never computes a hash.

#![allow(dead_code)] // each test file uses its own part of this module

use std::path::{Path, PathBuf};

use declared::DeclaredSection;
use diff::{DiffReport, Refusal, Side};

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// One compiled model: what a [`Side`] borrows from.
pub struct Compiled {
    pub name: String,
    pub declared: DeclaredSection,
    pub model_hash: String,
    pub compiler: Option<String>,
}

impl Compiled {
    pub fn read(path: &Path) -> Compiled {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let envelope = ir::envelope_from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        Compiled {
            name: path.file_name().expect("a file name").to_string_lossy().into_owned(),
            model_hash: runid::inputs::model_ir_hash(&envelope.model).to_hex(),
            compiler: envelope.validated_by.clone(),
            declared: envelope.declared,
        }
    }

    pub fn side(&self) -> Side<'_> {
        Side {
            name: &self.name,
            declared: &self.declared,
            model_hash: &self.model_hash,
            compiler: self.compiler.as_deref(),
        }
    }
}

pub fn compare(left: &Compiled, right: &Compiled) -> Result<DiffReport, Refusal> {
    diff::diff(&left.side(), &right.side())
}

/// A file of the fixture set `tests/fixtures/diff_pairs` (`make
/// update-diff-pairs-golden` compiles it; `make check-diff-pairs-golden` fails
/// when a compiled file drifts from its source).
pub fn pair_file(stem: &str) -> Compiled {
    Compiled::read(&repo_root().join("tests/fixtures/diff_pairs/ir").join(format!("{stem}.ir.json")))
}

/// Every compiled golden: `ocaml/golden` and each `tests/fixtures/*/ir`.
pub fn golden_files() -> Vec<PathBuf> {
    let root = repo_root();
    let mut dirs = vec![root.join("ocaml/golden")];
    for entry in std::fs::read_dir(root.join("tests/fixtures")).expect("read tests/fixtures") {
        let ir = entry.expect("a directory entry").path().join("ir");
        if ir.is_dir() {
            dirs.push(ir);
        }
    }
    let mut files: Vec<PathBuf> = dirs
        .iter()
        .flat_map(|d| std::fs::read_dir(d).unwrap_or_else(|e| panic!("read {}: {e}", d.display())))
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .collect();
    files.sort();
    files
}
