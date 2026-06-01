//! Reading the new-format (`runid::RunRecord`) CAS store.
//!
//! This is the generic, Layout-driven half of the transitional reader: a
//! data-driven walk where the presence of a parseable `RunRecord` `run.json`
//! is the only discovery signal — no hardcoded level depth. The legacy
//! `run_meta::Run` kinds (fit/profile/survey) keep their own discovery in
//! `browse.rs` until M3 migrates their writers; until then `browse` dispatches
//! by subtree (new `sims/` here, old kinds there).
//!
//! An old `run_meta::Run` never deserializes as a `RunRecord` (it lacks the
//! required `format_version`/`run_id`/`levels`/… fields), so walking a legacy
//! subtree through here simply finds nothing — there is no cross-format
//! mis-parse hazard.

use std::path::{Path, PathBuf};

use runid::{ArtifactKind, RunRecord};

/// Recursively collect every `(dir, RunRecord)` under `subtree` whose dir holds
/// a parseable `RunRecord` `run.json`. Hidden dirs (`.staging`, `.quarantine`)
/// are skipped. Descends through leaves too, so declared child sub-artifacts
/// (`obs/…`, added in M2.5) are discovered as their own records.
pub fn walk_records(subtree: &Path) -> Vec<(PathBuf, RunRecord)> {
    let mut out = Vec::new();
    if !subtree.exists() {
        return out;
    }
    let mut stack = vec![subtree.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rj = dir.join("run.json");
        if rj.is_file() {
            if let Ok(bytes) = std::fs::read(&rj) {
                if let Ok(rec) = serde_json::from_slice::<RunRecord>(&bytes) {
                    out.push((dir.clone(), rec));
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let name = e.file_name();
                if name.to_string_lossy().starts_with('.') {
                    continue; // .staging / .quarantine
                }
                stack.push(p);
            }
        }
    }
    out
}

/// A new-format leaf record with its directory, plus convenience accessors
/// that read the factored level labels (provenance) for display.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub dir: PathBuf,
    pub record: RunRecord,
}

impl Leaf {
    /// A level's readable label by level name (`"model"`, `"scenario"`, …).
    pub fn level_label(&self, name: &str) -> &str {
        self.record
            .levels
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.label.as_str())
            .unwrap_or("")
    }

    pub fn run_id_hex(&self) -> String {
        self.record.run_id.to_hex()
    }

    /// The base seed parsed from the `seed_{n}` label (0 if absent/unparsed).
    pub fn seed(&self) -> u64 {
        self.level_label("seed")
            .strip_prefix("seed_")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// `traj.tsv` size in bytes (0 if absent).
    pub fn traj_bytes(&self) -> u64 {
        std::fs::metadata(self.dir.join("traj.tsv")).map(|m| m.len()).unwrap_or(0)
    }
}

/// All `sims/` leaves of kind `Sim` (new-format trajectory runs).
pub fn walk_sim_leaves(root: &Path) -> Vec<Leaf> {
    walk_records(&root.join("sims"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::Sim)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
}

/// New-format sims whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution; combined with the legacy `run.hash` matches in
/// `browse::resolve_any` so a user can address any run during M2→M3).
pub fn resolve_sim_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    walk_sim_leaves(root)
        .into_iter()
        .filter(|s| s.run_id_hex().starts_with(prefix))
        .collect()
}
