//! The check that a completed folder is whole: every listed file present with
//! its recorded size and modification time, and nothing unlisted.
//!
//! The check passes over the store's own entries (`.staging/`, `.attempts/`,
//! the lock files, `identity.json`, `run.json` and their temporaries), over
//! the attempt files the record names, and over subdirectories that hold
//! nested results. A root-level subdirectory that no manifest entry lies under
//! holds nested results when every file below it lies inside a result folder
//! whose `identity.json` is of a registered kind and names this result as its
//! parent, or inside a folder
//! holding only the `.lock` an in-place claim writes first. Full digests are
//! not computed here; `verify` and every read through the store compute them.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::artifact::Kinds;
use crate::fsutil::{fmt_mtime, is_lock_entry, is_temporary_of, rel_key};
use crate::hash::Key;
use crate::record::{
    read_record, IdentityFile, Read, RunFile, ATTEMPTS_DIR, IDENTITY_FILE, LOCK_FILE, PROGRESS_FILE, RUN_FILE, RUN_TMP,
    STAGING_DIR,
};

/// What is wrong with a completed folder, or `None` when it is whole.
pub(crate) fn damage(dir: &Path, run: &RunFile, run_id: &Key, kinds: &Kinds) -> Option<String> {
    for (name, entry) in run.files.iter().filter(|(_, e)| e.in_manifest()) {
        let path = dir.join(name);
        let meta = match fs::metadata(&path) {
            Ok(m) if m.is_file() => m,
            _ => return Some(format!("the file `{name}` is missing")),
        };
        if Some(meta.len()) != entry.bytes {
            return Some(format!("the file `{name}` has changed size"));
        }
        match meta.modified() {
            Ok(t) if Some(fmt_mtime(t)) == entry.mtime => {}
            _ => return Some(format!("the file `{name}` has changed since it was stored")),
        }
    }
    let listed: BTreeSet<&str> = run.files.keys().map(String::as_str).collect();
    let mut under: BTreeSet<String> = BTreeSet::new();
    for name in run.files.keys() {
        let parts: Vec<&str> = name.split('/').collect();
        for i in 1..parts.len() {
            under.insert(parts[..i].join("/"));
        }
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => return Some(format!("its folder cannot be listed: {e}")),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_store_entry(&name, run) {
            continue;
        }
        if is_dir {
            if under.contains(&name) {
                if let Some(what) = unlisted_below(dir, &entry.path(), &listed, &under) {
                    return Some(what);
                }
            } else if !holds_nested_results(&entry.path(), run_id, kinds) {
                return Some(format!("the directory `{name}` is not part of the result"));
            }
        } else if !listed.contains(name.as_str()) {
            return Some(format!("the file `{name}` is not part of the result"));
        }
    }
    None
}

fn is_store_entry(name: &str, run: &RunFile) -> bool {
    matches!(name, IDENTITY_FILE | RUN_FILE | RUN_TMP | STAGING_DIR | ATTEMPTS_DIR)
        || is_lock_entry(name)
        || is_temporary_of(name, RUN_FILE)
        || is_temporary_of(name, IDENTITY_FILE)
        || is_temporary_of(name, PROGRESS_FILE)
        || run.files.get(name).is_some_and(|e| !e.in_manifest())
        || run.files.iter().any(|(f, e)| !e.in_manifest() && is_temporary_of(name, f))
}

/// An unlisted file or directory below `sub`, a directory the manifest's
/// files lie in.
fn unlisted_below(root: &Path, sub: &Path, listed: &BTreeSet<&str>, under: &BTreeSet<String>) -> Option<String> {
    let entries = fs::read_dir(sub).ok()?;
    for entry in entries.flatten() {
        let rel = rel_key(root, &entry.path());
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if !under.contains(&rel) {
                return Some(format!("the directory `{rel}` is not part of the result"));
            }
            if let Some(w) = unlisted_below(root, &entry.path(), listed, under) {
                return Some(w);
            }
        } else if !listed.contains(rel.as_str()) {
            return Some(format!("the file `{rel}` is not part of the result"));
        }
    }
    None
}

/// Whether every file below `dir` lies inside a result folder nested in the
/// result `parent`, or in a folder an in-place claim has just begun.
fn holds_nested_results(dir: &Path, parent: &Key, kinds: &Kinds) -> bool {
    let Ok(entries) = fs::read_dir(dir) else { return false };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            return false;
        }
        let path = entry.path();
        if path.join(IDENTITY_FILE).is_file() {
            match read_record::<IdentityFile>(&path.join(IDENTITY_FILE)) {
                Read::Ok(f) if kinds.has(&f.kind) && f.parent.as_ref().is_some_and(|p| &p.run_id == parent) => continue,
                _ => return false,
            }
        }
        if only_a_lock(&path) || holds_nested_results(&path, parent, kinds) {
            continue;
        }
        return false;
    }
    true
}

fn only_a_lock(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else { return false };
    let names: Vec<String> = entries.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.iter().any(|n| n == LOCK_FILE) && names.iter().all(|n| is_lock_entry(n) || is_temporary_of(n, IDENTITY_FILE))
}
