//! Importing results copied from another store.
//!
//! [`Store::import`] takes a partial copy of another store: its `store.json`,
//! the result folders wanted, each at the path relative to the root it had
//! there, with the parent folders a nested result sits in, and the invocation
//! records that listed them. It moves the copy into `<root>/.staging/`, on this
//! store's filesystem, checks every result folder in it, and places them only
//! when every one passes, each by one rename. It then records them in one
//! invocation of this store, with outcome `reused` and the attributes the copy's
//! invocation records gave them.
//!
//! A result keeps the path it had in the other store. Under
//! [`crate::TopLevel::ByBuild`] that path begins with the other build's
//! directory, named by its ambient key, so a result of another build never
//! sits among this build's results. A nested result whose parent this store
//! already holds goes to the same path relative to that parent's folder.
//!
//! The checks are those of `verify`, made on every byte: the run id computed
//! again from `identity.json`, every result and provenance file hashed and
//! compared with `run.json`, and nothing in a folder that its record does not
//! name besides the store's own entries and its nested results. A run id
//! already held here is compared by its result files: equal digests leave the
//! stored folder in place, and other digests are refused with
//! [`Error::ImportedDiffers`], as is a folder path held by another result.
//! Provenance files are not compared, as on a recompute.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::artifact::run_id;
use crate::check::{damage, only_a_lock};
use crate::error::{At, Error, Result};
use crate::fsutil::{
    copy_tree, create_dirs_durable, fsync_dir, is_lock_entry, is_temporary_of, import_draft_name, remove_all, rename_noreplace,
};
use crate::hash::{Key, HASH_VERSION};
use crate::invocation::{all_invocations, RunInfo};
use crate::maintenance::walk;
use crate::owner::this_owner;
use crate::record::{
    read_record, Failure, IdentityFile, Lineage, Listing, Outcome, Pending, Read, Record, Relation, Role, RunFile,
    Status, StoreFile, ATTEMPTS_DIR, IDENTITY_FILE, PROGRESS_FILE, RUN_FILE, RUN_TMP, STAGING_DIR, STORE_FORMAT,
};
use crate::storage::{folder_state, identity_of, visible_dirs, State, Store};
use crate::Reuse;

/// What an import did with each result of the copy, by run id, in the copy's
/// parent-before-child order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Imported {
    /// Moved into this store.
    pub placed: Vec<Key>,
    /// Held here already with the same result files, and left in place.
    pub present: Vec<Key>,
}

/// One result folder of the copy, read and checked.
struct Found {
    /// Its path relative to the copy's root.
    rel: PathBuf,
    ident: IdentityFile,
    run: RunFile,
    ambient: Key,
    attributes: serde_json::Value,
}

/// What placing does with one result of the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Held here with the same result files: left where it is.
    Present,
    /// Renamed on its own to its destination.
    Move,
    /// Carried to its destination by the rename of a result that holds it.
    Carried,
}

struct Planned {
    found: Found,
    action: Action,
    /// The folder the result has in this store once placed: its stored
    /// folder for [`Action::Present`].
    dest: PathBuf,
    /// The position, in the plan, of the result of the copy it sits inside.
    holder: Option<usize>,
    /// Its folder in the copy as given, for refusals.
    shown: PathBuf,
}

/// How the copy reached `.staging/`.
#[derive(Clone, Copy)]
enum Moved {
    Renamed,
    /// Copied across filesystems; the original is removed once every result
    /// is placed.
    Copied,
}

impl Store {
    /// Checks every result folder of the store copy at `copy`, then places
    /// them, then records them in `info`'s invocation. Nothing is placed
    /// unless every folder passes.
    ///
    /// The copy is moved into `<root>/.staging/` first, so each rename that
    /// places a folder is atomic. When the import succeeds the copy is gone:
    /// its results are in the store and the rest is deleted. When it is
    /// refused, or a placement or the record fails, every folder already
    /// placed is moved back into the copy, the copy is moved back to `copy`,
    /// and no invocation is recorded.
    ///
    /// Refused with [`Error::ImportRefused`], naming the folder and the
    /// reason, for a copy that is not a store of this store's format and
    /// layout, and for a result folder whose record is not completed, of a
    /// registered kind and of this record format; whose run id does not
    /// compute from its `identity.json`; whose files do not have their
    /// recorded digests or which holds a file its record does not name; which
    /// no invocation record in the copy lists; or whose parent, or a result it
    /// depends on, is neither in the copy nor completed in this store. Refused
    /// with [`Error::ImportedDiffers`] when this store holds a result's run id,
    /// or its folder's path, with other result files.
    pub fn import(&self, copy: &Path, info: &RunInfo) -> Result<Imported> {
        self.check_copy_root(copy)?;
        let staging_root = self.inner.root.join(STAGING_DIR);
        std::fs::create_dir_all(&staging_root).at("create directory", &staging_root)?;
        let staged = staging_root.join(import_draft_name(this_owner()));
        let moved = match std::fs::rename(copy, &staged) {
            Ok(()) => Moved::Renamed,
            Err(e) if e.kind() == ErrorKind::CrossesDevices => match copy_tree(copy, &staged) {
                Ok(()) => Moved::Copied,
                Err(e) => {
                    let _ = remove_all(&staged);
                    return Err(e);
                }
            },
            Err(e) => return Err(e).at("move into the store's staging", copy),
        };
        fsync_dir(&staging_root)?;
        let imported = self.import_staged(&staged, copy, info);
        match (&imported, moved) {
            (Ok(_), Moved::Renamed) => remove_all(&staged)?,
            (Ok(_), Moved::Copied) => {
                remove_all(&staged)?;
                remove_all(copy)?;
            }
            // A refused copy goes back to where it was given. Should that
            // fail, it stays in `.staging/`, which `clean` removes once this
            // process has stopped.
            (Err(_), Moved::Renamed) => {
                let _ = std::fs::rename(&staged, copy);
            }
            (Err(_), Moved::Copied) => {
                let _ = remove_all(&staged);
            }
        }
        imported
    }

    /// Refuses a copy that is not a directory holding a `store.json` of this
    /// store's format and layout, and the store's root or a directory holding
    /// it.
    fn check_copy_root(&self, copy: &Path) -> Result<()> {
        let refuse = |reason: String| Error::ImportRefused { folder: copy.to_path_buf(), reason };
        let meta = std::fs::symlink_metadata(copy).map_err(|e| refuse(format!("it cannot be read: {e}")))?;
        if !meta.is_dir() {
            return Err(refuse("it is not a directory".into()));
        }
        let canonical = copy.canonicalize().map_err(|e| refuse(format!("it cannot be read: {e}")))?;
        if let Ok(root) = self.inner.root.canonicalize() {
            if root.starts_with(&canonical) {
                return Err(refuse("it is this store, or holds it".into()));
            }
            if canonical.starts_with(&root) {
                return Err(refuse("it is inside this store".into()));
            }
        }
        let store_json = copy.join("store.json");
        let bytes = std::fs::read(&store_json).map_err(|e| refuse(format!("its store.json cannot be read: {e}")))?;
        let f: StoreFile = serde_json::from_slice(&bytes).map_err(|e| refuse(format!("its store.json does not read: {e}")))?;
        if f.format != STORE_FORMAT {
            return Err(refuse(format!("its store.json has format {}; this program reads format {STORE_FORMAT}", f.format)));
        }
        if f.top_level != self.inner.top.wire() {
            return Err(refuse(format!(
                "it lays results out {}, and this store lays them out {}",
                f.top_level,
                self.inner.top.wire()
            )));
        }
        Ok(())
    }

    /// Checks, places and records the copy at `staged`, under `.staging/`.
    /// `shown` is the path the copy was given as, for refusals.
    pub(crate) fn import_staged(&self, staged: &Path, shown: &Path, info: &RunInfo) -> Result<Imported> {
        self.refuse_unverifiable(staged, shown)?;
        let attributes = copy_attributes(staged);
        let mut found = Vec::new();
        for (rel, ident) in identity_files(staged) {
            let refuse = |reason: String| Error::ImportRefused { folder: shown.join(&rel), reason };
            let ident = match ident {
                Read::Ok(f) => f,
                Read::Bad(e) => return Err(refuse(e.to_string())),
                Read::Absent => continue,
            };
            let (run, ambient) = self.check_folder(&staged.join(&rel), &ident).map_err(refuse)?;
            let Some(attributes) = attributes.get(&ident.run_id).cloned() else {
                return Err(refuse("no invocation record in the copy lists it as completed".into()));
            };
            found.push(Found { rel, ident, run, ambient, attributes });
        }
        if found.is_empty() {
            return Err(Error::ImportRefused { folder: shown.to_path_buf(), reason: "it holds no result folder".into() });
        }
        found.sort_by(|a, b| a.rel.cmp(&b.rel));
        let planned = self.plan_import(found, shown)?;
        for p in planned.iter().filter(|p| p.action != Action::Present) {
            drop_owned_entries(&staged.join(&p.found.rel))?;
        }
        sync_tree(staged)?;

        // Every folder is placed before anything is recorded, and a failure
        // at any later step moves every placed folder back into the staged
        // copy, so an import either places and records the whole copy or
        // leaves the store as it was.
        let mut placing = Placing::default();
        let done = match self.place(staged, &planned, &mut placing) {
            Ok(done) => done,
            Err(e) => {
                placing.undo();
                return Err(e);
            }
        };
        let recorded = self.record(&planned, &done, info);
        if recorded.is_err() {
            placing.undo();
        }
        recorded
    }

    /// Records the placed copy: one invocation entry per result, then one
    /// index line per result placed. An entry is `reused` once every step has
    /// succeeded, and `failed` otherwise.
    fn record(&self, planned: &[Planned], done: &[Action], info: &RunInfo) -> Result<Imported> {
        let inner = &self.inner;
        let entries = planned
            .iter()
            .map(|p| Listing {
                kind: p.found.ident.kind.clone(),
                run_id: p.found.ident.run_id,
                attributes: p.found.attributes.clone(),
                source: None,
                outcome: Pending::Planned,
            })
            .collect();
        let idx = info.shared.register(inner, entries, Reuse::IfCompleted)?;
        let indexed = (|| -> Result<Imported> {
            let mut out = Imported::default();
            for (p, action) in planned.iter().zip(done) {
                let key = p.found.ident.run_id;
                if *action == Action::Present {
                    out.present.push(key);
                    continue;
                }
                let upstream = p.found.ident.parent.as_ref().filter(|q| q.relation == Relation::Upstream).map(|q| q.run_id);
                inner.index_completed(&p.found.ident.kind, key, &p.dest, &p.found.run, upstream)?;
                out.placed.push(key);
            }
            Ok(out)
        })();
        match &indexed {
            Ok(_) => info.shared.set(&idx, Outcome::Reused),
            Err(e) => {
                let failure = Failure { operation: Some("importing".into()), reason: Some(e.to_string()) };
                info.shared.set(&idx, Outcome::Failed(failure));
            }
        }
        indexed
    }

    /// The checks of one result folder of the copy, as `verify` makes them,
    /// on every byte. Returns its record and its ambient key, or the reason it
    /// is refused.
    fn check_folder(&self, dir: &Path, ident: &IdentityFile) -> std::result::Result<(RunFile, Key), String> {
        let kinds = &self.inner.kinds;
        if !kinds.has(&ident.kind) {
            return Err(format!("its kind `{}` is not one this program stores", ident.kind));
        }
        let run: RunFile = match read_record(&dir.join(RUN_FILE)) {
            Read::Ok(r) => r,
            Read::Absent => return Err("it has no run.json".into()),
            Read::Bad(e) => return Err(e.to_string()),
        };
        if run.status != Status::Completed {
            return Err(format!("its run.json says {}, not completed", status_word(run.status)));
        }
        if ident.hash_version != HASH_VERSION {
            return Err(format!("its run id is of hash version {}; this program computes version {HASH_VERSION}", ident.hash_version));
        }
        let identity_key = kinds.record_key(&ident.kind, &ident.identity).map_err(|e| e.to_string())?;
        let Some(ambient) = self.inner.ambient_key_of(&ident.ambient) else {
            return Err("its ambient identity does not read as this program's".into());
        };
        let computed = run_id(&ident.kind, &ambient, &identity_key);
        if computed != ident.run_id {
            return Err(format!("its identity.json computes to the run id {computed}, not {}", ident.run_id));
        }
        let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let (short, full) = (format!("-{}", ident.run_id.short12()), format!("-{}", ident.run_id.to_hex()));
        if !name.ends_with(&short) && !name.ends_with(&full) {
            return Err(format!("its folder's name does not end in its run id, `{short}`"));
        }
        for (file, entry) in run.files.iter().filter(|(_, e)| e.in_manifest()) {
            let path = dir.join(file);
            match std::fs::symlink_metadata(&path) {
                Ok(m) if m.is_file() => {}
                Ok(_) => return Err(format!("`{file}` is not a regular file")),
                Err(e) => return Err(format!("the file `{file}` cannot be read: {e}")),
            }
            let Some(recorded) = entry.sha256 else { return Err(format!("run.json records no digest for `{file}`")) };
            match file_digest(&path) {
                Ok(d) if d == recorded => {}
                Ok(_) => return Err(format!("the file `{file}` does not have its recorded digest")),
                Err(e) => return Err(format!("the file `{file}` cannot be read: {e}")),
            }
        }
        if let Some(what) = damage(dir, &run, &ident.run_id, kinds) {
            return Err(what);
        }
        Ok((run, ambient))
    }

    /// Decides where each result goes, refusing before anything moves.
    fn plan_import(&self, found: Vec<Found>, shown: &Path) -> Result<Vec<Planned>> {
        let inner = &self.inner;
        let held: HashMap<Key, (PathBuf, IdentityFile)> = walk(inner).into_iter().map(|(d, f)| (f.run_id, (d, f))).collect();
        let in_copy: BTreeSet<Key> = found.iter().map(|f| f.ident.run_id).collect();
        let completed_here = |key: &Key| {
            held.get(key).is_some_and(|(d, _)| matches!(folder_state(d, key, &inner.kinds), State::Completed { .. }))
        };
        let mut planned: Vec<Planned> = Vec::new();
        for f in found {
            let imported = shown.join(&f.rel);
            let refuse = |reason: String| Error::ImportRefused { folder: imported.clone(), reason };
            let key = f.ident.run_id;
            // The nearest result of the copy this one sits inside.
            let holder_at = planned
                .iter()
                .enumerate()
                .filter(|(_, p)| f.rel.starts_with(&p.found.rel))
                .max_by_key(|(_, p)| p.found.rel.components().count())
                .map(|(i, _)| i);
            let holder = holder_at.map(|i| &planned[i]);
            let (dest, carried) = match (&f.ident.parent, holder) {
                (Some(parent), Some(h)) => {
                    if h.found.ident.run_id != parent.run_id || h.found.ident.kind != parent.kind {
                        return Err(refuse(format!(
                            "it sits inside the {} {} and names the {} {} as its parent",
                            h.found.ident.kind,
                            h.found.ident.run_id.short12(),
                            parent.kind,
                            parent.run_id.short12()
                        )));
                    }
                    let below = f.rel.strip_prefix(&h.found.rel).expect("a holder's path is a prefix");
                    (h.dest.join(below), h.action != Action::Present)
                }
                (Some(parent), None) => {
                    let Some(at) = parent_folder(&f.rel, &parent.run_id) else {
                        return Err(refuse(format!("it sits in no folder of its parent, the {} {}", parent.kind, parent.run_id.short12())));
                    };
                    let local = match held.get(&parent.run_id) {
                        Some((d, p)) if p.kind == parent.kind => d,
                        _ => {
                            return Err(refuse(format!(
                                "its parent, the {} {}, is neither in the copy nor in this store",
                                parent.kind,
                                parent.run_id.short12()
                            )))
                        }
                    };
                    if !matches!(folder_state(local, &parent.run_id, &inner.kinds), State::Completed { .. }) {
                        return Err(refuse(format!("its parent is held in this store at {}, not completed", local.display())));
                    }
                    (local.join(f.rel.strip_prefix(&at).expect("an ancestor")), false)
                }
                (None, Some(h)) => {
                    return Err(refuse(format!(
                        "it sits inside the {} {} and names no parent",
                        h.found.ident.kind,
                        h.found.ident.run_id.short12()
                    )))
                }
                (None, None) => {
                    if !self.is_result_path(&f.rel, &f.ident.kind, &f.ambient) {
                        return Err(refuse(format!("it is not at a path where this store places a result of kind `{}`", f.ident.kind)));
                    }
                    (inner.root.join(&f.rel), false)
                }
            };
            let (action, dest) = match held.get(&key) {
                Some((local, _)) => {
                    if carried {
                        return Err(refuse(format!(
                            "this store holds it at {}, outside the folder of its parent, which it does not hold",
                            local.display()
                        )));
                    }
                    match folder_state(local, &key, &inner.kinds) {
                        State::Completed { run, .. } if result_digests(&run) == result_digests(&f.run) => (Action::Present, local.clone()),
                        State::Completed { .. } => {
                            return Err(Error::ImportedDiffers { run_id: key, local: local.clone(), imported });
                        }
                        other => {
                            return Err(refuse(format!("this store holds it at {}, {}", local.display(), state_words(&other))));
                        }
                    }
                }
                None if carried => (Action::Carried, dest),
                None => {
                    if std::fs::symlink_metadata(&dest).is_ok() {
                        return Err(taken(key, &dest, &imported));
                    }
                    (Action::Move, dest)
                }
            };
            if action != Action::Present {
                if let Some((kind, id)) = upstreams(&f.run).find(|(_, id)| !in_copy.contains(id) && !completed_here(id)) {
                    return Err(refuse(format!(
                        "its dependency, the {kind} {}, is neither in the copy nor in this store",
                        id.short12()
                    )));
                }
            }
            planned.push(Planned { found: f, action, dest, holder: holder_at, shown: imported });
        }
        Ok(planned)
    }

    /// Whether `rel` is where this store's layout puts a result that sits in
    /// no other: under `ByBuild`, `<label>-<12 hex of its ambient key>/<kind>/…`,
    /// and under `ByKind`, `<kind>/…`.
    fn is_result_path(&self, rel: &Path, kind: &str, ambient: &Key) -> bool {
        let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        match &self.inner.top {
            crate::TopLevel::ByBuild { .. } => {
                parts.len() >= 3 && parts[0].ends_with(&format!("-{}", ambient.short12())) && parts[1] == kind
            }
            crate::TopLevel::ByKind => parts.len() >= 2 && parts[0] == kind,
        }
    }

    /// Renames each result that moves on its own into place, parents first,
    /// and returns what became of each: moved, carried by its holder's
    /// rename, or present.
    ///
    /// A destination that another writer filled since the plan, such as a
    /// second import of the same result, is compared: a completed result
    /// with the same result files makes this one present, and anything else is
    /// refused with [`Error::ImportedDiffers`]. A result whose holder turned
    /// out present is then renamed on its own into the holder's folder.
    fn place(&self, staged: &Path, planned: &[Planned], placing: &mut Placing) -> Result<Vec<Action>> {
        let mut done: Vec<Action> = planned.iter().map(|p| p.action).collect();
        for (i, p) in planned.iter().enumerate() {
            let moves = match p.action {
                Action::Present => false,
                Action::Move => true,
                Action::Carried => {
                    let h = p.holder.expect("a carried result has a holder");
                    done[h] == Action::Present
                }
            };
            if !moves {
                continue;
            }
            #[cfg(test)]
            tests::before_move(&p.dest)?;
            let parent = p.dest.parent().expect("a destination under the root");
            placing.made.extend(missing_dirs(&self.inner.root, parent));
            create_dirs_durable(&self.inner.root, parent)?;
            let src = staged.join(&p.found.rel);
            match rename_noreplace(&src, &p.dest) {
                Ok(()) => {
                    placing.moved.push((src, p.dest.clone()));
                    fsync_dir(parent)?;
                    done[i] = Action::Move;
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists || p.dest.exists() => {
                    let key = p.found.ident.run_id;
                    match folder_state(&p.dest, &key, &self.inner.kinds) {
                        State::Completed { run, .. } if result_digests(&run) == result_digests(&p.found.run) => {
                            done[i] = Action::Present;
                        }
                        _ => return Err(taken(key, &p.dest, &p.shown)),
                    }
                }
                Err(e) => return Err(e).at("move the imported result into place", &p.dest),
            }
        }
        Ok(done)
    }
}

/// What `place` has done so far, so that it can be undone.
#[derive(Default)]
struct Placing {
    /// Each folder renamed into place: where it was in the staged copy, and
    /// where it went.
    moved: Vec<(PathBuf, PathBuf)>,
    /// Each directory the placements created, a parent before its children.
    made: Vec<PathBuf>,
}

impl Placing {
    /// Moves every placed folder back to where it was in the staged copy,
    /// last first, and removes each directory the placements created that
    /// then holds nothing; one another writer has put something in since is
    /// kept. No index line or invocation record names a placed folder yet. A
    /// folder that cannot be moved back stays where it was placed, whole.
    fn undo(self) {
        for (src, dest) in self.moved.iter().rev() {
            let _ = std::fs::rename(dest, src);
        }
        for dir in self.made.iter().rev() {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// The ancestors of `dir`, `dir` included and `root` excluded, that do not
/// exist, the highest first.
fn missing_dirs(root: &Path, dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut cur = Some(dir);
    while let Some(d) = cur {
        if d == root || d.exists() {
            break;
        }
        out.push(d.to_path_buf());
        cur = d.parent();
    }
    out.reverse();
    out
}

/// The refusal of a result whose place in this store is taken: by a result,
/// which then holds other bytes, or by a folder that holds none.
fn taken(run_id: Key, dest: &Path, imported: &Path) -> Error {
    match identity_of(dest) {
        Some(_) => Error::ImportedDiffers { run_id, local: dest.to_path_buf(), imported: imported.to_path_buf() },
        None => Error::ImportRefused {
            folder: imported.to_path_buf(),
            reason: format!("its place in this store, {}, is taken by a folder that holds no result", dest.display()),
        },
    }
}

impl Store {
    /// Refuses a copy holding a symbolic link anywhere, or, in a result
    /// folder, a dot-named directory other than the store's own `.staging/`
    /// and `.attempts/`: the checks never read below one, so its contents
    /// would be placed unverified.
    fn refuse_unverifiable(&self, staged: &Path, shown: &Path) -> Result<()> {
        fn visit(dir: &Path, staged: &Path, shown: &Path, in_result: bool) -> Result<()> {
            for entry in std::fs::read_dir(dir).at("list", dir)?.flatten() {
                let path = entry.path();
                let rel = path.strip_prefix(staged).expect("below the copy").to_path_buf();
                let refuse = |reason: &str| Error::ImportRefused { folder: shown.join(&rel), reason: reason.into() };
                let kind = std::fs::symlink_metadata(&path).at("stat", &path)?.file_type();
                if kind.is_symlink() {
                    return Err(refuse("it is a symbolic link"));
                }
                if !kind.is_dir() {
                    continue;
                }
                let name = entry.file_name().to_string_lossy().into_owned();
                let dot = name.starts_with('.');
                if dot && in_result && name != STAGING_DIR && name != ATTEMPTS_DIR {
                    return Err(refuse("it is a dot-named directory inside a result, which no check reads"));
                }
                let result_here = in_result || (!dot && path.join(IDENTITY_FILE).exists());
                visit(&path, staged, shown, result_here)?;
            }
            Ok(())
        }
        visit(staged, staged, shown, false)
    }
}

/// The attributes each result has in the copy: those the latest ended invocation
/// record that lists it as computed or reused gave it.
fn copy_attributes(copy: &Path) -> BTreeMap<Key, serde_json::Value> {
    let mut out = BTreeMap::new();
    for inv in all_invocations(copy).0 {
        let Record::Ended(closed) = inv else { continue };
        for r in closed.results.into_iter().filter(|r| r.outcome.completed()) {
            out.insert(r.run_id, r.attributes);
        }
    }
    out
}

/// Every visible folder below `root` that holds an `identity.json`, by its
/// path relative to `root`, with that file as read.
fn identity_files(root: &Path) -> Vec<(PathBuf, Read<IdentityFile>)> {
    fn visit(dir: &Path, root: &Path, out: &mut Vec<(PathBuf, Read<IdentityFile>)>) {
        for child in visible_dirs(dir) {
            let file = child.join(IDENTITY_FILE);
            if file.exists() {
                out.push((child.strip_prefix(root).expect("below the root").to_path_buf(), read_record(&file)));
            }
            visit(&child, root, out);
        }
    }
    let mut out = Vec::new();
    visit(root, root, &mut out);
    out
}

/// The deepest folder on `rel`'s path, above it, whose name ends in the run
/// id `parent`: where the parent's folder was in the other store.
fn parent_folder(rel: &Path, parent: &Key) -> Option<PathBuf> {
    let (short, full) = (format!("-{}", parent.short12()), format!("-{}", parent.to_hex()));
    let mut at = rel.parent();
    while let Some(p) = at {
        if let Some(Component::Normal(n)) = p.components().next_back() {
            let n = n.to_string_lossy();
            if n.ends_with(&short) || n.ends_with(&full) {
                return Some(p.to_path_buf());
            }
        }
        at = p.parent();
    }
    None
}

/// Every stored result a record names as an upstream, by kind and run id:
/// its dependencies, the results its lineage records without hashing (such
/// as a fit run's quantities), and the results it read while it ran.
fn upstreams(run: &RunFile) -> impl Iterator<Item = (&str, &Key)> {
    let deps = run.deps.iter().map(|d| (d.kind.as_str(), &d.run_id));
    let lineage = run.lineage.iter().filter_map(|l| match l {
        Lineage::Result { kind, run_id, .. } => Some((kind.as_str(), run_id)),
        Lineage::Input { .. } => None,
    });
    let reads = run.reads.iter().map(|r| (r.kind.as_str(), &r.run_id));
    deps.chain(lineage).chain(reads)
}

/// A record's result files, by name, with their digests.
fn result_digests(run: &RunFile) -> BTreeMap<&str, Option<Key>> {
    run.files.iter().filter(|(_, e)| e.role == Role::Result).map(|(n, e)| (n.as_str(), e.sha256)).collect()
}

fn status_word(s: Status) -> &'static str {
    match s {
        Status::Running => "running",
        Status::Completed => "completed",
        Status::Failed => "failed",
    }
}

fn state_words(s: &State) -> String {
    match s {
        State::Damaged { what, .. } => format!("damaged: {what}"),
        State::Failed { .. } => "failed".into(),
        State::HalfMade { owner: None, .. } => "in a folder with no record".into(),
        _ => "running".into(),
    }
}

/// The SHA-256 of the file at `path`, read in pieces.
fn file_digest(path: &Path) -> std::io::Result<Key> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
    Ok(Key::from_bytes(hasher.finalize().into()))
}

/// Removes from an imported result folder, at every level, the entries that
/// belong to the other store's processes rather than to the result, none of
/// which the checks read: `.staging/`, where a process built by-products;
/// `.attempts/`, the records of earlier attempts; lock files, and folders
/// holding only the lock an in-place claim writes first; and the temporaries
/// of the records and the progress file. Left in place, a lock or a claim
/// naming an owner on another host would read here as a claim no process on
/// this host can take, and the rest would be bytes no digest covers.
fn drop_owned_entries(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir).at("list", dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            if name == STAGING_DIR || name == ATTEMPTS_DIR || only_a_lock(&path) {
                remove_all(&path)?;
            } else {
                drop_owned_entries(&path)?;
            }
        } else if is_lock_entry(&name)
            || name == RUN_TMP
            || [IDENTITY_FILE, RUN_FILE, PROGRESS_FILE].iter().any(|f| is_temporary_of(&name, f))
        {
            remove_all(&path)?;
        }
    }
    Ok(())
}

/// Syncs every file and directory below `dir`, and `dir`, so a folder renamed
/// out of it is never a durable entry pointing at data that has not reached
/// the disk.
fn sync_tree(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir).at("list", dir)?.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() => sync_tree(&path)?,
            Ok(t) if t.is_file() => std::fs::File::open(&path).and_then(|f| f.sync_all()).at("sync", &path)?,
            _ => {}
        }
    }
    fsync_dir(dir)
}

#[cfg(test)]
mod tests;
