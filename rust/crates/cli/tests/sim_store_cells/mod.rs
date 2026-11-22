//! Reads the simulation cells `camdl simulate` stores in a run store.
//!
//! A cell is a folder under `<root>/<build>/sim/…` holding `identity.json`
//! (kind `sim`, its run id and identity record), `run.json` and its result
//! files. Cells are found by their `identity.json`, never by parsing folder
//! names.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One stored simulation cell.
#[derive(Debug, Clone)]
pub struct SimCell {
    /// The cell folder.
    pub dir: PathBuf,
    /// The cell's run id, from `identity.json`.
    pub run_id: String,
    /// The parsed `identity.json`.
    pub identity: serde_json::Value,
}

impl SimCell {
    /// The cell's `run.json`, parsed.
    pub fn run_json(&self) -> serde_json::Value {
        read_json(&self.dir.join("run.json"))
    }

    /// The cell's `traj.tsv` as text.
    pub fn traj(&self) -> String {
        std::fs::read_to_string(self.dir.join("traj.tsv"))
            .unwrap_or_else(|e| panic!("read {}/traj.tsv: {e}", self.dir.display()))
    }

    /// The identity record's `config` table.
    pub fn config(&self) -> &serde_json::Value {
        &self.identity["identity"]["config"]
    }

    /// The identity record's parameter values.
    pub fn params(&self) -> &serde_json::Value {
        &self.identity["identity"]["params"]["values"]
    }

    /// The active scenario names in the identity record.
    pub fn scenario_names(&self) -> Vec<String> {
        self.identity["identity"]["scenario"]["names"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default()
    }

    /// The process seed in the identity record.
    pub fn seed(&self) -> u64 {
        self.identity["identity"]["seed"]["seed"]
            .as_u64()
            .unwrap_or_else(|| panic!("cell {} has no seed", self.dir.display()))
    }

    /// The name of the scenario folder the cell sits in (`_none_` for the
    /// implicit scenario).
    pub fn scenario_dir_name(&self) -> String {
        let name = self
            .dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        strip_hash(name).to_string()
    }

    /// The name of the params folder the cell sits in (`base`, or the
    /// sweep/draw label).
    pub fn params_dir_name(&self) -> String {
        let name = self
            .dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        strip_hash(name).to_string()
    }

    /// The name of the config folder the cell sits in, `<backend>-dt<dt>`.
    pub fn config_dir_name(&self) -> String {
        let name = self
            .dir
            .ancestors()
            .nth(3)
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("");
        strip_hash(name).to_string()
    }

    /// The by-product files of the cell whose file name is `name`, at any
    /// depth below the cell folder.
    pub fn by_products(&self, name: &str) -> Vec<PathBuf> {
        walk_files(&self.dir)
            .into_iter()
            .filter(|p| p.file_name().and_then(|s| s.to_str()) == Some(name))
            .collect()
    }
}

/// `<label>-<12hex>` → `<label>`.
fn strip_hash(name: &str) -> &str {
    match name.rsplit_once('-') {
        Some((head, tail)) if tail.len() == 12 && tail.chars().all(|c| c.is_ascii_hexdigit()) => {
            head
        }
        _ => name,
    }
}

/// Parses a JSON file, panicking with its path on failure.
pub fn read_json(p: &Path) -> serde_json::Value {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

/// Every regular file below `root`, sorted.
pub fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Every live stored simulation cell under the store `root`, sorted by
/// folder. The store's own dot-folders (`.quarantine`, `.staging`, …) hold no
/// live cells and are not searched.
pub fn sim_cells(root: &Path) -> Vec<SimCell> {
    let mut cells: Vec<SimCell> = walk_files(root)
        .into_iter()
        .filter(|p| {
            !p.strip_prefix(root)
                .unwrap_or(p)
                .components()
                .any(|c| c.as_os_str().to_str().is_some_and(|s| s.starts_with('.')))
        })
        .filter(|p| p.file_name().and_then(|s| s.to_str()) == Some("identity.json"))
        .filter_map(|p| {
            let v = read_json(&p);
            if v["kind"] != "sim" {
                return None;
            }
            let run_id = v["run_id"].as_str()?.to_string();
            Some(SimCell { dir: p.parent().unwrap().to_path_buf(), run_id, identity: v })
        })
        .collect();
    cells.sort_by(|a, b| a.dir.cmp(&b.dir));
    cells
}

/// The one stored simulation cell under `root`; panics unless exactly one.
pub fn only_sim_cell(root: &Path) -> SimCell {
    let cells = sim_cells(root);
    assert_eq!(
        cells.len(),
        1,
        "expected exactly one sim cell under {}, got {:?}",
        root.display(),
        cells.iter().map(|c| &c.dir).collect::<Vec<_>>()
    );
    cells.into_iter().next().unwrap()
}

/// The sorted run ids of every stored simulation cell under `root`.
pub fn sim_run_ids(root: &Path) -> Vec<String> {
    let mut ids: Vec<String> = sim_cells(root).into_iter().map(|c| c.run_id).collect();
    ids.sort();
    ids
}
