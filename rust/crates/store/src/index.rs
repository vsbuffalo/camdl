//! The index: a set of append-only logs, one per owner, under `.index/`.
//!
//! A `finish`, and the end of an invocation, append one line to the owner's
//! log. Readers merge the logs. The index is a cache: everything in it can be
//! rebuilt from the records and the invocation records, and a folder it names
//! is confirmed against that folder's `identity.json` before it is used.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{At, Result};
use crate::hash::Key;
use crate::owner::this_owner;
use crate::record::IndexLine;

pub(crate) struct Index {
    folders: HashMap<Key, String>,
    /// Dependency edges, both ways: what a result consumed, and what consumed it.
    pub(crate) consumed: HashMap<Key, Vec<Key>>,
    pub(crate) consumers: HashMap<Key, Vec<Key>>,
    log: Option<File>,
}

pub(crate) fn index_dir(root: &Path) -> PathBuf {
    root.join(".index")
}

pub(crate) fn own_log(root: &Path) -> PathBuf {
    index_dir(root).join(format!("{}.log", this_owner().file_stem()))
}

impl Index {
    /// Merges every log under the root. A line that does not parse is left
    /// out, so a log a crash cut short loses only its last line.
    pub(crate) fn load(root: &Path) -> Index {
        let mut idx = Index { folders: HashMap::new(), consumed: HashMap::new(), consumers: HashMap::new(), log: None };
        let Ok(entries) = std::fs::read_dir(index_dir(root)) else { return idx };
        let mut logs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "log")).collect();
        logs.sort();
        for log in logs {
            let Ok(text) = std::fs::read_to_string(&log) else { continue };
            for line in text.lines() {
                if let Ok(l) = serde_json::from_str::<IndexLine>(line) {
                    idx.learn(&l);
                }
            }
        }
        idx
    }

    fn learn(&mut self, line: &IndexLine) {
        if let IndexLine::Completed { run_id, folder, deps, lineage, upstream, .. } = line {
            self.folders.insert(*run_id, folder.clone());
            let ups: Vec<Key> = deps.iter().chain(lineage).chain(upstream.iter()).copied().collect();
            for u in &ups {
                self.consumers.entry(*u).or_default().push(*run_id);
            }
            self.consumed.insert(*run_id, ups);
        }
    }

    /// The folder the index records for `run_id`, relative to the root.
    pub(crate) fn folder(&self, run_id: &Key) -> Option<&str> {
        self.folders.get(run_id).map(String::as_str)
    }

    /// Appends one line to this owner's log.
    pub(crate) fn append(&mut self, root: &Path, line: &IndexLine) -> Result<()> {
        if self.log.is_none() {
            let dir = index_dir(root);
            std::fs::create_dir_all(&dir).at("create directory", &dir)?;
            let path = own_log(root);
            self.log = Some(OpenOptions::new().create(true).append(true).open(&path).at("open", &path)?);
        }
        let mut bytes = serde_json::to_vec(line).expect("an index line encodes");
        bytes.push(b'\n');
        let path = own_log(root);
        self.log.as_mut().expect("opened above").write_all(&bytes).at("append to", &path)?;
        self.learn(line);
        Ok(())
    }

    /// Forgets the open log, after it was replaced by a compacted one.
    pub(crate) fn reopen(&mut self) {
        self.log = None;
    }
}
