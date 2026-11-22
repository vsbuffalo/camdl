//! Walking every result, and `clean`, `verify` and `reindex`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::artifact::run_id;
use crate::error::Result;
use crate::fsutil::{draft_owner, fsync_dir, remove_all, write_file_atomic, write_json_atomic};
use crate::hash::Key;
use crate::index::{index_dir, own_log, Index};
use crate::invocation::{all_invocations, invocation_path};
use crate::owner::{local_liveness, local_liveness_of_stem, Liveness, Owner, Timestamp};
use crate::record::{
    read_record, Failure, IdentityFile, IndexLine, InvocationStatus, Outcome, Read, Role, RunFile, Status, ATTEMPTS_DIR,
    RECORD_FORMAT, RUN_FILE, STAGING_DIR,
};
use crate::storage::{folder_state, identity_of, visible_dirs, Inner, State, Store};

/// Every folder under the root's result trees that holds an `identity.json`,
/// nested results included, with that file.
pub(crate) fn walk(inner: &Inner) -> Vec<(PathBuf, IdentityFile)> {
    let mut out = Vec::new();
    for top in inner.all_tops() {
        for kind_dir in visible_dirs(&top) {
            visit(&kind_dir, &mut out);
        }
    }
    out
}

fn visit(dir: &Path, out: &mut Vec<(PathBuf, IdentityFile)>) {
    for child in visible_dirs(dir) {
        if let Some(f) = identity_of(&child) {
            out.push((child.clone(), f));
        }
        visit(&child, out);
    }
}

/// Replaces this owner's log with one line per completed result found and per
/// ended invocation, and removes the logs of owners known not to be running.
pub(crate) fn write_compacted_index(inner: &Inner, found: &[(PathBuf, IdentityFile)]) -> Result<()> {
    if inner.scratch {
        return Ok(());
    }
    let mut idx = inner.index.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for (dir, f) in found {
        let State::Completed { run, .. } = folder_state(dir, &f.run_id, &inner.kinds) else { continue };
        let line = IndexLine::Completed {
            format: RECORD_FORMAT,
            kind: f.kind.clone(),
            run_id: f.run_id,
            folder: crate::fsutil::rel_key(&inner.root, dir),
            invocation: run.invocation,
            deps: run.deps.iter().map(|d| d.run_id).collect(),
            lineage: run
                .lineage
                .iter()
                .filter_map(|l| match l {
                    crate::record::Lineage::Result { run_id, .. } => Some(*run_id),
                    crate::record::Lineage::Input { .. } => None,
                })
                .collect(),
            upstream: f.parent.as_ref().filter(|p| p.relation == crate::record::Relation::Upstream).map(|p| p.run_id),
        };
        text.push_str(&serde_json::to_string(&line).expect("an index line encodes"));
        text.push('\n');
    }
    for inv in all_invocations(&inner.root).into_iter().filter(|i| i.status == InvocationStatus::Ended) {
        let line = IndexLine::Invocation {
            format: RECORD_FORMAT,
            seq: inv.seq,
            label: inv.label.clone(),
            results: inv.results.iter().map(|r| r.run_id).collect(),
        };
        text.push_str(&serde_json::to_string(&line).expect("an index line encodes"));
        text.push('\n');
    }
    let dir = index_dir(&inner.root);
    std::fs::create_dir_all(&dir).map_err(|e| crate::Error::Io { op: "create directory", path: dir.clone(), source: e })?;
    let mine = own_log(&inner.root);
    write_file_atomic(&mine, text.as_bytes())?;
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path == mine || path.extension().is_none_or(|x| x != "log") {
            continue;
        }
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if owner_not_running(inner, &stem) {
            let _ = std::fs::remove_file(&path);
        }
    }
    let mut fresh = Index::load(&inner.root);
    fresh.reopen();
    *idx = Some(fresh);
    Ok(())
}

/// Whether the owner whose files are named `stem` is known not to be
/// running: it has no beat file, or its beat file names a process dead on
/// this host.
fn owner_not_running(inner: &Inner, stem: &str) -> bool {
    let beat = inner.root.join(".owners").join(format!("{stem}.json"));
    match read_record::<crate::owner::BeatFile>(&beat) {
        Read::Absent => true,
        Read::Ok(b) => {
            let owner = Owner { host: b.host, pid: b.pid, started: b.started };
            owner.is_this_host() && local_liveness(&owner) == Liveness::Dead
        }
        Read::Bad(_) => false,
    }
}

/// One result `verify` found wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyFailure {
    pub folder: PathBuf,
    pub run_id: Key,
    pub what: String,
}

/// What `clean` removed, and the staging it left because its owner is on
/// another host.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cleaned {
    pub staging: Vec<PathBuf>,
    pub quarantine: Vec<PathBuf>,
    pub attempts: Vec<PathBuf>,
    pub directories: Vec<PathBuf>,
    /// Claims, and folders a claim was being built in, whose owner is a
    /// process on another host: this host cannot tell whether it has stopped.
    pub other_hosts: Vec<PathBuf>,
    /// The sequence numbers of the invocations whose records it closed as
    /// interrupted.
    pub invocations: Vec<u64>,
}

impl Store {
    /// Checks every completed result, or the ones given: computes each run
    /// id again from `identity.json`, and each manifest file's SHA-256. Writes
    /// nothing.
    pub fn verify(&self, only: Option<&[PathBuf]>) -> Result<Vec<VerifyFailure>> {
        let inner = &self.inner;
        let found: Vec<(PathBuf, IdentityFile)> = match only {
            Some(dirs) => dirs.iter().filter_map(|d| identity_of(d).map(|f| (d.clone(), f))).collect(),
            None => walk(inner),
        };
        let mut out = Vec::new();
        for (dir, f) in found {
            let fail = |what: String| VerifyFailure { folder: dir.clone(), run_id: f.run_id, what };
            let run: RunFile = match read_record(&dir.join(RUN_FILE)) {
                Read::Ok(r) => r,
                Read::Absent => continue,
                Read::Bad(e) => {
                    out.push(fail(e.to_string()));
                    continue;
                }
            };
            if run.status != Status::Completed {
                continue;
            }
            let ident_key = match inner.kinds.record_key(&f.kind, &f.identity) {
                Ok(k) => k,
                Err(e) => {
                    out.push(fail(e.to_string()));
                    continue;
                }
            };
            let Some(ambient) = inner.ambient_key_of(&f.ambient) else {
                out.push(fail("its ambient identity does not read".into()));
                continue;
            };
            let computed = run_id(&f.kind, &ambient, &ident_key);
            if computed != f.run_id {
                out.push(fail(format!("its identity.json computes to the run id {computed}")));
                continue;
            }
            for (name, e) in run.files.iter().filter(|(_, e)| e.role != Role::Attempt) {
                match std::fs::read(dir.join(name)) {
                    Ok(b) if Some(Key::digest_bytes(&b)) == e.sha256 => {}
                    Ok(_) => out.push(fail(format!("the file `{name}` does not have its recorded digest"))),
                    Err(err) => out.push(fail(format!("the file `{name}` cannot be read: {err}"))),
                }
            }
        }
        Ok(out)
    }

    /// Deletes staging folders whose owner is dead on this host (claims, and
    /// the folders claims were being built in), everything in
    /// `.quarantine/`, every `.attempts/` folder, and every empty group, kind
    /// and build directory, and closes the record of each running invocation
    /// whose owner is dead on this host as interrupted. Deletes no result and
    /// touches no completed result, running claim or index, and leaves
    /// staging whose owner is on another host, listing it in
    /// [`Cleaned::other_hosts`].
    pub fn clean(&self) -> Result<Cleaned> {
        let inner = &self.inner;
        let mut out = Cleaned::default();
        let mut staging_dirs = vec![inner.root.join(STAGING_DIR)];
        let found = walk(inner);
        for (dir, _) in &found {
            staging_dirs.push(dir.join(STAGING_DIR));
        }
        for s in staging_dirs {
            for entry in std::fs::read_dir(&s).into_iter().flatten().flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                // A folder a claim was being built in, before its rename: its
                // name says whose it is, and only its builder ever touches it.
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(stem) = draft_owner(&name) {
                    match local_liveness_of_stem(stem) {
                        Some(Liveness::Dead) => {
                            remove_all(&path)?;
                            out.staging.push(path);
                        }
                        Some(_) => {}
                        None => out.other_hosts.push(path),
                    }
                    continue;
                }
                let owner = match read_record::<RunFile>(&path.join(RUN_FILE)) {
                    Read::Ok(r) => r.owner,
                    _ => continue,
                };
                if !owner.is_this_host() {
                    out.other_hosts.push(path);
                } else if inner.liveness(&owner, Duration::ZERO) == Liveness::Dead && inner.clear_claim(&path, &owner, false)? {
                    out.staging.push(path);
                }
            }
        }
        for mut inv in all_invocations(&inner.root).into_iter().filter(|i| i.status == InvocationStatus::Running) {
            if !inv.owner.is_this_host() || inner.liveness(&inv.owner, Duration::ZERO) != Liveness::Dead {
                continue;
            }
            for r in inv.results.iter_mut().filter(|r| r.outcome == Outcome::Planned) {
                r.outcome = Outcome::Failed;
                r.failure = Some(Failure {
                    operation: None,
                    reason: Some("the command was interrupted before this result ended".into()),
                });
                inv.counts.failed += 1;
            }
            inv.status = InvocationStatus::Interrupted;
            inv.ended = Some(Timestamp::now());
            write_json_atomic(&invocation_path(&inner.root, inv.seq), &inv)?;
            let _ = std::fs::remove_file(inner.root.join(".invocations").join(format!("{}.progress.json", inv.seq)));
            out.invocations.push(inv.seq);
        }
        let q = inner.root.join(".quarantine");
        for entry in std::fs::read_dir(&q).into_iter().flatten().flatten() {
            remove_all(&entry.path())?;
            out.quarantine.push(entry.path());
        }
        for (dir, _) in &found {
            let a = dir.join(ATTEMPTS_DIR);
            if a.is_dir() {
                remove_all(&a)?;
                out.attempts.push(a);
            }
        }
        for top in inner.all_tops() {
            for kind_dir in visible_dirs(&top) {
                prune_empty(&kind_dir, &mut out.directories)?;
            }
            if top != inner.root && is_empty(&top) {
                std::fs::remove_dir(&top).ok();
                out.directories.push(top);
            }
        }
        Ok(out)
    }

    /// Rebuilds and compacts the index from the records and the invocation
    /// records.
    pub fn reindex(&self) -> Result<usize> {
        let found = walk(&self.inner);
        write_compacted_index(&self.inner, &found)?;
        Ok(found.len())
    }
}

/// Removes every directory below `dir` that holds no file and no result,
/// then `dir` itself if it is then empty. A result folder is never removed.
fn prune_empty(dir: &Path, removed: &mut Vec<PathBuf>) -> Result<()> {
    if identity_of(dir).is_some() || dir.join(crate::record::LOCK_FILE).exists() {
        return Ok(());
    }
    for child in visible_dirs(dir) {
        prune_empty(&child, removed)?;
    }
    if is_empty(dir) {
        std::fs::remove_dir(dir).ok();
        if let Some(p) = dir.parent() {
            let _ = fsync_dir(p);
        }
        removed.push(dir.to_path_buf());
    }
    Ok(())
}

fn is_empty(dir: &Path) -> bool {
    std::fs::read_dir(dir).map(|mut d| d.next().is_none()).unwrap_or(false)
}

#[cfg(test)]
mod tests;
