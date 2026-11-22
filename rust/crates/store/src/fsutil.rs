//! Durable writes, lock files and the quarantine.
//!
//! Every write is made durable in one order: each file synced, then the
//! record, then its folder, then the rename, then the destination's parent. A
//! `rename` without those syncs is not crash-atomic: a durable directory
//! entry can point at data that never reached the disk.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{json_err, At, Error, Result};
use crate::owner::{this_owner, Liveness, Owner};

/// `"{secs}.{nanos:09}"` since the Unix epoch: the manifest's form of a
/// modification time, compared exactly.
pub(crate) fn fmt_mtime(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:09}", d.as_secs(), d.subsec_nanos())
}

pub(crate) fn fsync_dir(path: &Path) -> Result<()> {
    File::open(path).and_then(|f| f.sync_all()).at("sync directory", path)
}

pub(crate) fn write_file_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = File::create(path).at("create", path)?;
    f.write_all(bytes).at("write", path)?;
    f.sync_all().at("sync", path)
}

/// A name no other writer uses: `<prefix>.<pid>.<n>.<nanos>`.
pub(crate) fn unique_name(prefix: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}.{}.{n}.{nanos}", std::process::id())
}

/// The name of the temporary folder a claim of the run id `hex` is built in
/// under `.staging/` before it is renamed to `hex`:
/// `<hex>.<host>-<pid>-<start>.<n>`, naming the owner that builds it, so that
/// `clean` can tell whether that owner has stopped ([`draft_owner`]).
pub(crate) fn claim_draft_name(owner: &Owner, hex: &str) -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!("{hex}.{}.{}", owner.file_stem(), N.fetch_add(1, Ordering::Relaxed))
}

/// The owner-name part of a `.staging/` entry named by [`claim_draft_name`],
/// or `None` for any other name.
pub(crate) fn draft_owner(name: &str) -> Option<&str> {
    let (hex, rest) = name.split_at_checked(64)?;
    if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let (stem, n) = rest.strip_prefix('.')?.rsplit_once('.')?;
    n.parse::<u64>().ok()?;
    Some(stem)
}

/// Writes `bytes` to `path` so a reader sees the old file or the whole new
/// one: through a temporary file of this writer's own in the same directory,
/// renamed over `path`, and the directory synced.
pub(crate) fn write_file_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{}.tmp", unique_name(&name)));
    let committed = write_file_synced(&tmp, bytes).and_then(|()| fs::rename(&tmp, path).at("replace", path));
    if let Err(e) = committed {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(dir) => fsync_dir(dir),
        None => fsync_dir(Path::new(".")),
    }
}

/// Writes `value` as pretty JSON to `path`, durably.
pub(crate) fn write_json_synced<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| json_err("encode", path, e))?;
    write_file_synced(path, &bytes)
}

/// Replaces `path` with `value` as pretty JSON, atomically.
pub(crate) fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| json_err("encode", path, e))?;
    write_file_atomic(path, &bytes)
}

/// Creates `dir` and every missing ancestor below `stop`, syncing each one
/// made and its parent, so a folder renamed under them is never under a
/// directory a crash could lose. An existing directory is not an error.
pub(crate) fn create_dirs_durable(stop: &Path, dir: &Path) -> Result<()> {
    let mut missing = Vec::new();
    let mut cur = dir.to_path_buf();
    while !cur.exists() && cur != stop {
        missing.push(cur.clone());
        match cur.parent() {
            Some(p) => cur = p.to_path_buf(),
            None => break,
        }
    }
    for d in missing.iter().rev() {
        match fs::create_dir(d) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).at("create directory", d),
        }
        fsync_dir(d)?;
        if let Some(p) = d.parent() {
            fsync_dir(p)?;
        }
    }
    Ok(())
}

/// What a lock file holds: `.lock` in an in-place result folder,
/// `.invocations/.lock`, and the `.reclaim` serialiser beside a `.lock`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LockFile {
    pub format: u16,
    pub owner: Owner,
}

pub(crate) const LOCK_FORMAT: u16 = 1;

/// Reads the owner a lock file names. `Ok(None)` when there is no lock file.
pub(crate) fn lock_owner(path: &Path) -> Result<Option<Owner>> {
    match fs::read(path) {
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).at("read lock", path),
        Ok(bytes) => serde_json::from_slice::<LockFile>(&bytes)
            .map(|l| Some(l.owner))
            .map_err(|e| Error::Parse { what: path.display().to_string(), why: e.to_string() }),
    }
}

/// A held lock file, removed when dropped.
#[derive(Debug)]
pub(crate) struct LockGuard {
    path: PathBuf,
}

impl LockGuard {
    /// Gives up the guard without removing the file, for a lock that moved
    /// into the quarantine with its folder: by the time the guard would drop,
    /// the old path may hold another claimant's lock.
    pub(crate) fn forget(self) {
        std::mem::forget(self);
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// The outcome of trying to take a lock file.
pub(crate) enum LockAttempt {
    Acquired(LockGuard),
    /// Held by an owner that is not known to be dead, or by this process.
    Held(Owner, Liveness),
}

/// Creates `path` holding `contents`, failing with `AlreadyExists` if it
/// exists. The contents are written to a temporary file first and linked into
/// place, so a lock file is never seen empty or half-written.
fn create_exclusive(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{}.tmp", unique_name(&name)));
    {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(contents)?;
        f.sync_all()?;
    }
    let linked = fs::hard_link(&tmp, path);
    let _ = fs::remove_file(&tmp);
    linked
}

fn lock_contents() -> Vec<u8> {
    serde_json::to_vec(&LockFile { format: LOCK_FORMAT, owner: this_owner().clone() })
        .expect("a lock file encodes")
}

/// Tries once to take the lock file at `path`.
///
/// A lock whose owner `liveness` reports dead is taken over without the file
/// ever being absent: the taker first creates the `.reclaim` serialiser beside
/// it, checks that the lock still names the dead owner, and renames a file
/// naming itself over it. While the dead lock exists, every other taker's
/// exclusive create fails and meets the serialiser, so two takers never both
/// replace it. A serialiser left by a dead taker is removed and the takeover
/// tried again.
pub(crate) fn try_lock(path: &Path, liveness: &dyn Fn(&Owner) -> Liveness) -> Result<LockAttempt> {
    for _ in 0..8 {
        match create_exclusive(path, &lock_contents()) {
            Ok(()) => return Ok(LockAttempt::Acquired(LockGuard { path: path.to_path_buf() })),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e).at("create lock", path),
        }
        let Some(holder) = lock_owner(path)? else { continue };
        if &holder == this_owner() {
            return Ok(LockAttempt::Held(holder, Liveness::Alive));
        }
        let l = liveness(&holder);
        if l != Liveness::Dead {
            return Ok(LockAttempt::Held(holder, l));
        }
        if take_over(path, &holder, liveness)? {
            return Ok(LockAttempt::Acquired(LockGuard { path: path.to_path_buf() }));
        }
    }
    let holder = lock_owner(path)?.unwrap_or_else(|| this_owner().clone());
    Ok(LockAttempt::Held(holder, Liveness::Unknown))
}

/// Replaces the lock at `path`, held by the dead `holder`, with one naming
/// this process. `Ok(false)` when another taker got there first.
fn take_over(path: &Path, holder: &Owner, liveness: &dyn Fn(&Owner) -> Liveness) -> Result<bool> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let reclaim = path.with_file_name(format!("{name}.reclaim"));
    match create_exclusive(&reclaim, &lock_contents()) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            let stranded = match lock_owner(&reclaim) {
                Ok(Some(o)) => &o != this_owner() && liveness(&o) == Liveness::Dead,
                Ok(None) => false,
                Err(_) => false,
            };
            if stranded {
                let _ = fs::remove_file(&reclaim);
            }
            return Ok(false);
        }
        Err(e) => return Err(e).at("create lock", &reclaim),
    }
    let outcome = (|| -> Result<bool> {
        if lock_owner(path)?.as_ref() != Some(holder) {
            return Ok(false);
        }
        let tmp = path.with_file_name(format!(".{}.tmp", unique_name(&name)));
        write_file_synced(&tmp, &lock_contents())?;
        fs::rename(&tmp, path).at("take over lock", path)?;
        Ok(true)
    })();
    let _ = fs::remove_file(&reclaim);
    outcome
}

/// Whether `name` is one of the files [`try_lock`] keeps beside a lock.
pub(crate) fn is_lock_entry(name: &str) -> bool {
    name == ".lock" || name == ".lock.reclaim" || (name.starts_with("..lock.") && name.ends_with(".tmp"))
}

/// Whether `name` is the temporary file of an atomic replacement of `file`
/// ([`write_file_atomic`]), left behind by a writer that stopped midway.
pub(crate) fn is_temporary_of(name: &str, file: &str) -> bool {
    name.starts_with(&format!(".{file}.")) && name.ends_with(".tmp")
}

/// Moves `dir` into `<root>/.quarantine/` under its own name, with `.1`, `.2`
/// appended when that name is taken, and returns where it went.
/// A folder that is already gone, moved by another process a moment before,
/// is not an error.
pub(crate) fn quarantine(root: &Path, dir: &Path) -> Result<PathBuf> {
    let q = root.join(".quarantine");
    fs::create_dir_all(&q).at("create directory", &q)?;
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "result".into());
    let mut dest = q.join(&name);
    let mut n = 0;
    while dest.exists() {
        n += 1;
        dest = q.join(format!("{name}.{n}"));
    }
    match fs::rename(dir, &dest) {
        Err(e) if e.kind() == ErrorKind::NotFound && !dir.exists() => return Ok(dest),
        r => r.at("move into the quarantine", dir)?,
    }
    fsync_dir(&q)?;
    Ok(dest)
}

/// The `/`-joined path of `file` relative to `root`.
pub(crate) fn rel_key(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Removes `path`, a file or a directory tree; an absent path is not an error.
pub(crate) fn remove_all(path: &Path) -> Result<()> {
    let r = match fs::symlink_metadata(path) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).at("stat", path),
        Ok(m) if m.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
    };
    match r {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(e).at("remove", path),
        _ => Ok(()),
    }
}
