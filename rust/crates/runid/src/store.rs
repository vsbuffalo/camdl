//! `CasStore` — lookup and the atomic/durable commit protocol.
//!
//! Two commit modes:
//!
//! - **Mode A — atomic stage-then-rename** (sim, obs, pfilter, survey,
//!   profile-point — single-shot): write the whole leaf into a staging dir
//!   on the *same filesystem* as `results/`, apply the fsync ordering, then
//!   `rename` it into place. A reader never sees a half-written leaf.
//! - **Mode B — streamed `Running → Completed`** (fit stages — long,
//!   streaming, resumable): claim the leaf dir exclusively via an `O_EXCL`
//!   `.lock`, write a `Running` `run.json`, stream files, then commit by
//!   `run.json.tmp → rename` over `run.json`.
//!
//! **Durability is explicit code.** `rename` without a barrier is *not*
//! crash-atomic — a durable dir entry can point at an inode whose data
//! blocks never flushed. Both modes implement: write each artifact then
//! `sync_all`; write `run.json` then `sync_all`; `sync_all` the containing
//! dir; `rename`; `sync_all` the parent of the destination.
//!
//! **Path existence never implies identity.** Before any clear or overwrite,
//! the incumbent `run.json`'s full `run_id` is compared to the expected
//! identity: a match is safe to clear/recompute; a mismatch is a
//! [`Lookup::Collision`] (PathPrefixCollision) — a *different* artifact
//! occupies a short-hash-colliding path, and it is disambiguated
//! (`{seg}` → `{seg}~{hash16}` → `{seg}~{full64}`), never deleted.
//!
//! ## M1 scope note
//!
//! The dead-PID reclaim of a Mode B claim ("a `Running` `run.json` whose
//! lock-holder PID is dead is a reclaimable stale claim") needs a process
//! liveness check. M1 keeps runid's dependency surface to `ir` only and
//! defers liveness to M3 (which owns the fit/resume rewrite and its
//! concurrent-fit gate). Until then Mode B is *conservative*: a held `.lock`
//! is always `FitInProgress` (never auto-reclaimed), and a stale `Running`
//! leaf with **no** `.lock` is reclaimed by clearing its orphan contents
//! under a fresh exclusive lock.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::hash::{ContentHash, HASH_VERSION};
use crate::record::{FileChecksum, RunRecord, RunStatus, FORMAT_VERSION};

/// The identity a lookup/commit expects to find at a path. Identity is the
/// `run_id` (a function of kind + level hashes); a path may hold a *different*
/// identity that shares the 8-char short-hash segment (a PathPrefixCollision).
#[derive(Debug, Clone)]
pub struct LeafIdentity {
    pub run_id: ContentHash,
}

impl LeafIdentity {
    pub fn new(run_id: ContentHash) -> Self {
        Self { run_id }
    }
}

/// The outcome of a lookup at a path.
///
/// `RunRecord` is ~376 bytes; the record-carrying variants are boxed so the
/// enum stays small (clippy `large_enum_variant`, a hard error under the
/// repo's `-D warnings`). Callers match `Hit(record)`/`Collision(record)`
/// where `record: Box<RunRecord>` derefs transparently.
#[derive(Debug)]
pub enum Lookup {
    /// Identity matches, `Completed`, exact-set integrity ok.
    Hit(Box<RunRecord>),
    /// Nothing usable at the path (no `run.json`).
    Miss,
    /// SAME identity present but unusable → safe-clear + recompute.
    Stale(StaleReason),
    /// A DIFFERENT full identity occupies this path (short-hash collision) →
    /// disambiguate, never touch the incumbent.
    Collision(Box<RunRecord>),
}

/// Why a same-identity leaf is unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    /// `status != Completed` (a crashed/in-flight streamed run), or a
    /// `run.json` that does not parse.
    Incomplete,
    /// A listed file is missing or its size/mtime no longer matches.
    Corrupt,
    /// An unlisted file (not a declared child) is present.
    OrphanFiles,
    /// `hash_version`/`format_version` is not current.
    SchemaDrift,
}

/// Errors from the commit protocol.
#[derive(Debug, thiserror::Error)]
pub enum CasError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("checksum mismatch for {file}")]
    ChecksumMismatch { file: String },
    #[error("orphan files in leaf at {path}")]
    OrphanFiles { path: String },
    #[error("schema drift at {path}")]
    SchemaDrift { path: String },
    /// The Mode B `O_EXCL` claim is held (a concurrent fit, or — until M3's
    /// liveness check — a crashed one).
    #[error("fit in progress at {path} (pid {pid})")]
    FitInProgress { path: String, pid: u32 },
    /// A `claim_streaming` found a `Completed` same-identity leaf — the caller
    /// should have taken the cache hit from `lookup` instead of claiming.
    #[error("artifact already completed at {path}")]
    AlreadyCompleted { path: String },
}

/// A bundle of named output files (e.g. `{traj.tsv, event_log.tsv}`).
#[derive(Debug, Clone, Default)]
pub struct Artifacts {
    pub files: BTreeMap<String, Vec<u8>>,
}

impl Artifacts {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, name: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.files.insert(name.into(), bytes.into());
    }
}

/// The store contract. `lookup` reports cache validity; `commit` is the
/// durable Mode-A path. Mode B (streaming) has a different shape and lives on
/// the concrete [`FsCasStore`] as `claim_streaming` → [`StreamClaim`].
///
/// `commit` returns the *actual* destination path: a PathPrefixCollision may
/// land the leaf at a disambiguated sibling, which the caller must know to
/// address it. (This refines the proposal's `Result<(), CasError>`.)
pub trait CasStore {
    fn lookup(&self, path: &Path, expected: &LeafIdentity) -> Lookup;
    fn commit(
        &self,
        path: &Path,
        record: RunRecord,
        artifacts: Artifacts,
    ) -> Result<PathBuf, CasError>;
}

/// Process-local counter making each Mode-A staging dir unique per attempt,
/// so concurrent same-identity commits never share (and clobber) a staging
/// dir. Combined with the pid, the staging name is unique across processes.
static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

/// A filesystem-backed CAS rooted at a `results/` directory.
pub struct FsCasStore {
    root: PathBuf,
}

impl FsCasStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn staging_root(&self) -> PathBuf {
        self.root.join(".staging")
    }

    fn quarantine_root(&self) -> PathBuf {
        self.root.join(".quarantine")
    }

    /// Tiered-integrity lookup. Identity gate first (run.json present and its
    /// `run_id` equals `expected`; a present-but-different `run_id` is a
    /// PathPrefixCollision), then the cheap gate (`Completed`, current
    /// versions, exact-set manifest).
    pub fn lookup(&self, path: &Path, expected: &LeafIdentity) -> Lookup {
        let record = match read_record(path) {
            ReadResult::Absent => return Lookup::Miss,
            // A run.json that exists but does not parse is corruption (e.g. a
            // truncated write), not absence. Its identity is unknown, so the
            // commit path quarantines rather than blind-clears.
            ReadResult::Unparseable => return Lookup::Stale(StaleReason::Corrupt),
            ReadResult::Ok(r) => r,
        };
        // Identity gate.
        if !record.identity_matches(&expected.run_id) {
            return Lookup::Collision(record);
        }
        // Cheap gate.
        if record.hash_version != HASH_VERSION || record.format_version != FORMAT_VERSION {
            return Lookup::Stale(StaleReason::SchemaDrift);
        }
        if record.status != RunStatus::Completed {
            return Lookup::Stale(StaleReason::Incomplete);
        }
        match check_exact_set(path, &record) {
            ExactSet::Ok => Lookup::Hit(record),
            ExactSet::Corrupt => Lookup::Stale(StaleReason::Corrupt),
            ExactSet::Orphan => Lookup::Stale(StaleReason::OrphanFiles),
        }
    }

    /// Mode A: write the leaf into staging with the fsync ordering, then
    /// rename into place — collision-aware, never clobbering a different
    /// identity. Returns the destination (possibly disambiguated).
    pub fn commit_atomic(
        &self,
        path: &Path,
        mut record: RunRecord,
        artifacts: Artifacts,
    ) -> Result<PathBuf, CasError> {
        // Staging is unique **per attempt** — `{run_id}.{pid}.{nonce}` — so
        // two concurrent commits of the *same* identity (the dedup'd-draw-row
        // case under batch's rayon) never share a staging dir and clobber each
        // other's in-flight writes. The proposal's `.staging/{run_id}` named
        // the intent ("unique staging dir, race-safe by rename"); the pid +
        // process-local counter realize it. Orphaned `.staging/*` from a crash
        // is harmless debris swept at store-open (M2's store lifecycle).
        let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
        let staging = self
            .staging_root()
            .join(format!("{}.{}.{}", record.run_id.to_hex(), std::process::id(), nonce));
        // The unique path can only pre-exist as a stale orphan from a prior
        // process that reused our pid+nonce — never a live concurrent attempt —
        // so clearing it is safe.
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        fs::create_dir_all(&staging)?;

        // Write artifacts + build the exact-set manifest, fsync each file.
        let mut manifest = BTreeMap::new();
        for (name, bytes) in &artifacts.files {
            let fp = staging.join(name);
            if let Some(parent) = fp.parent() {
                fs::create_dir_all(parent)?;
            }
            write_file_synced(&fp, bytes)?;
            let meta = fs::metadata(&fp)?;
            manifest.insert(
                name.clone(),
                FileChecksum {
                    bytes: bytes.len() as u64,
                    mtime: fmt_mtime(meta.modified()?),
                    digest: ContentHash::digest_bytes(bytes),
                },
            );
        }
        record.artifacts = manifest;
        record.status = RunStatus::Completed;

        // run.json then fsync the staging dir.
        write_record_direct(&staging, &record)?;
        fsync_dir(&staging)?;

        let id = LeafIdentity::new(record.run_id);
        self.rename_staged_into_place(&staging, path, &id)
    }

    /// Resolve the destination and rename `staging` into it, tolerating the
    /// concurrent-rename race: a competing winner can create `dest` between
    /// our `resolve` and our `rename`, so a failed rename whose `dest` now
    /// exists re-resolves (the next lookup sees the winner's `Completed` leaf
    /// → benign Hit). This is the proposal's "if `final` exists when the
    /// rename is attempted, run `lookup(final, expected)`" made race-safe.
    fn rename_staged_into_place(
        &self,
        staging: &Path,
        path: &Path,
        id: &LeafIdentity,
    ) -> Result<PathBuf, CasError> {
        // The loop converges fast: a competing same-identity commit resolves
        // to AlreadyCompleted, a different identity disambiguates to its own
        // candidate. The cap is a defensive backstop, never reached in
        // practice.
        const MAX_ATTEMPTS: u32 = 64;
        for _ in 0..MAX_ATTEMPTS {
            match self.resolve_claim_dir(path, id)? {
                ClaimOutcome::AlreadyCompleted(dest) => {
                    // Lost a benign race: the identical leaf already landed.
                    fs::remove_dir_all(staging).ok();
                    return Ok(dest);
                }
                ClaimOutcome::Reclaim(dest) => {
                    // A concurrent thread may reclaim the same stale leaf;
                    // tolerate it already being gone.
                    if let Err(e) = fs::remove_dir_all(&dest) {
                        if e.kind() != ErrorKind::NotFound {
                            return Err(e.into());
                        }
                    }
                    if self.try_rename(staging, &dest)? {
                        return Ok(dest);
                    }
                }
                ClaimOutcome::Fresh(dest) => {
                    if self.try_rename(staging, &dest)? {
                        return Ok(dest);
                    }
                }
            }
        }
        Err(CasError::Io(std::io::Error::other(
            "commit did not converge: destination contended past retry cap",
        )))
    }

    /// Attempt `rename(staging, dest)`. Returns `Ok(true)` on success (and
    /// fsyncs the dest's parent), `Ok(false)` if a concurrent winner created
    /// `dest` first (caller re-resolves), or an error for any other failure.
    fn try_rename(&self, staging: &Path, dest: &Path) -> Result<bool, CasError> {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::rename(staging, dest) {
            Ok(()) => {
                if let Some(parent) = dest.parent() {
                    fsync_dir(parent)?;
                }
                Ok(true)
            }
            // A non-empty `dest` (a competing winner's leaf) makes `rename`
            // fail; that is the benign race, re-resolved by the caller. Any
            // other failure (dest still absent) is a real error.
            Err(e) => {
                if dest.exists() {
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    }

    /// Mode B begin: claim the leaf dir exclusively and write a `Running`
    /// `run.json`. Collision-aware (disambiguates). Fails fast with
    /// [`CasError::FitInProgress`] if a `.lock` is already held.
    pub fn claim_streaming(
        &self,
        path: &Path,
        running: RunRecord,
    ) -> Result<StreamClaim, CasError> {
        let id = LeafIdentity::new(running.run_id);
        let dir = match self.resolve_claim_dir(path, &id)? {
            ClaimOutcome::AlreadyCompleted(d) => {
                return Err(CasError::AlreadyCompleted { path: d.display().to_string() });
            }
            ClaimOutcome::Fresh(d) | ClaimOutcome::Reclaim(d) => d,
        };
        fs::create_dir_all(&dir)?;

        // The O_EXCL claim is the authoritative race guard.
        let lock = dir.join(".lock");
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(mut f) => {
                write!(f, "{}", std::process::id())?;
                f.sync_all()?;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                let pid = fs::read_to_string(&lock)
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                return Err(CasError::FitInProgress { path: dir.display().to_string(), pid });
            }
            Err(e) => return Err(e.into()),
        }

        // We now hold the exclusive lock. Clear any stale orphan contents
        // (a prior crashed run's chain files + stale run.json), preserving
        // only our fresh `.lock`.
        clear_except_lock(&dir)?;

        let mut rec = running;
        rec.status = RunStatus::Running;
        rec.artifacts = BTreeMap::new();
        write_record_atomic(&dir, &rec)?;
        fsync_dir(&dir)?;
        Ok(StreamClaim { dir })
    }

    /// Walk the disambiguation candidates for `path`, classifying each by a
    /// lookup so a different identity is never cleared. Returns the dir to
    /// claim (after quarantining an orphaned partial), or that the identical
    /// artifact is already present.
    fn resolve_claim_dir(
        &self,
        path: &Path,
        expected: &LeafIdentity,
    ) -> Result<ClaimOutcome, CasError> {
        for cand in disambiguation_candidates(path, &expected.run_id) {
            if !cand.exists() {
                return Ok(ClaimOutcome::Fresh(cand));
            }
            match self.lookup(&cand, expected) {
                Lookup::Hit(_) => return Ok(ClaimOutcome::AlreadyCompleted(cand)),
                Lookup::Stale(_) => {
                    // lookup only returns Stale after the identity gate passed
                    // (run_id matched) OR for an unparseable run.json. Clear
                    // only on a verified same-identity match; otherwise the
                    // identity is unknown → quarantine, never blind-clear.
                    if same_identity_on_disk(&cand, &expected.run_id) {
                        return Ok(ClaimOutcome::Reclaim(cand));
                    }
                    self.quarantine(&cand)?;
                    return Ok(ClaimOutcome::Fresh(cand));
                }
                // A different full identity shares this short-hash segment:
                // advance to the disambiguated candidate, never touch it.
                Lookup::Collision(_) => continue,
                // Dir exists with no run.json: an orphaned partial → quarantine.
                Lookup::Miss => {
                    self.quarantine(&cand)?;
                    return Ok(ClaimOutcome::Fresh(cand));
                }
            }
        }
        // The full-64 candidate is unique, so the loop always terminates with
        // a Claim; this is unreachable.
        unreachable!("disambiguation escalates to the unique full-64 segment")
    }

    /// Move an unclaimable leaf aside for manual repair rather than clearing
    /// it (its identity is unknown or it is an orphaned partial).
    fn quarantine(&self, cand: &Path) -> Result<(), CasError> {
        let q = self.quarantine_root();
        fs::create_dir_all(&q)?;
        let name = cand.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "leaf".into());
        let mut dest = q.join(&name);
        let mut n = 0;
        while dest.exists() {
            n += 1;
            dest = q.join(format!("{name}.{n}"));
        }
        fs::rename(cand, &dest)?;
        Ok(())
    }
}

impl CasStore for FsCasStore {
    fn lookup(&self, path: &Path, expected: &LeafIdentity) -> Lookup {
        FsCasStore::lookup(self, path, expected)
    }
    fn commit(
        &self,
        path: &Path,
        record: RunRecord,
        artifacts: Artifacts,
    ) -> Result<PathBuf, CasError> {
        self.commit_atomic(path, record, artifacts)
    }
}

/// A held Mode B claim: the exclusive lock on a leaf dir while it streams.
/// Stream files with [`StreamClaim::write`]; commit with
/// [`StreamClaim::finalize`].
#[derive(Debug)]
pub struct StreamClaim {
    dir: PathBuf,
}

impl StreamClaim {
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Stream one output file into the claimed leaf, fsync'd.
    pub fn write(&self, name: &str, bytes: &[u8]) -> Result<(), CasError> {
        let fp = self.dir.join(name);
        if let Some(parent) = fp.parent() {
            fs::create_dir_all(parent)?;
        }
        write_file_synced(&fp, bytes)?;
        Ok(())
    }

    /// Commit `Running → Completed`: build the exact-set manifest from the
    /// streamed files, write `run.json.tmp → rename` over `run.json` (the
    /// single-file rename is the commit point), fsync the dir, drop the lock.
    pub fn finalize(self, mut record: RunRecord) -> Result<PathBuf, CasError> {
        // Walk the whole streamed subtree (chains nest under `chain_N/`),
        // stopping at the declared children (`trajectories/`, `dt_check/`,
        // `obs/`) — those are separate artifacts, manifested by their own
        // commit, never folded into this leaf's exact set.
        record.artifacts = build_manifest(&self.dir, &record.children)?;
        record.status = RunStatus::Completed;
        write_record_atomic(&self.dir, &record)?;
        fsync_dir(&self.dir)?;
        fs::remove_file(self.dir.join(".lock")).ok();
        Ok(self.dir)
    }
}

// ── Free helpers ─────────────────────────────────────────────────────────────

enum ClaimOutcome {
    /// The identical artifact is already `Completed` here.
    AlreadyCompleted(PathBuf),
    /// A same-identity stale leaf is present → clear and recompute.
    Reclaim(PathBuf),
    /// The dir is absent (or was quarantined) → ready to claim.
    Fresh(PathBuf),
}

enum ReadResult {
    Absent,
    Unparseable,
    Ok(Box<RunRecord>),
}

enum ExactSet {
    Ok,
    /// A listed file is missing or size/mtime mismatched.
    Corrupt,
    /// An unlisted, non-child entry is present.
    Orphan,
}

/// Files the exact-set check and manifest builder ignore: the record itself,
/// its tmp, and the Mode B lock.
fn is_reserved(name: &str) -> bool {
    matches!(name, "run.json" | "run.json.tmp" | ".lock")
}

fn read_record(dir: &Path) -> ReadResult {
    match fs::read(dir.join("run.json")) {
        Err(_) => ReadResult::Absent,
        Ok(bytes) => match serde_json::from_slice::<RunRecord>(&bytes) {
            Ok(r) => ReadResult::Ok(Box::new(r)),
            Err(_) => ReadResult::Unparseable,
        },
    }
}

fn same_identity_on_disk(cand: &Path, run_id: &ContentHash) -> bool {
    matches!(read_record(cand), ReadResult::Ok(r) if &r.run_id == run_id)
}

/// The `/`-joined path of `file` relative to the leaf `root` — the stable,
/// OS-independent manifest key for a (possibly nested) own-file.
fn rel_key(root: &Path, file: &Path) -> String {
    file.strip_prefix(root)
        .unwrap_or(file)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// The leaf's OWN files must exactly match the manifest — no missing and no
/// orphan — across its **whole subtree**, EXCEPT the reserved files and the
/// declared `children` subdirs. A fit stage nests per-chain files
/// (`chain_1/trace.tsv`) plus stage-root outputs (`fit_state.toml`,
/// `draws.tsv`), so the walk recurses; but it **stops at a declared child**
/// (`obs/`, `trajectories/`, `dt_check/`) — those are separate artifacts with
/// their own `run.json`, validated on their own lookup, never recursed into
/// nor folded into this leaf's manifest.
///
/// This is the **cheap gate**: presence + size + mtime only, no digest, so a
/// fit with many/large chain files is not re-hashed on every `list`/`lookup`.
/// The "never serve wrong bytes" digest check runs at consume time.
fn check_exact_set(path: &Path, record: &RunRecord) -> ExactSet {
    // 1. Every listed file present at recorded bytes + mtime (keys may be
    //    nested, e.g. `chain_1/trace.tsv`; `Path::join` resolves the `/`).
    for (name, ck) in &record.artifacts {
        let fp = path.join(name);
        let meta = match fs::metadata(&fp) {
            Ok(m) if m.is_file() => m,
            _ => return ExactSet::Corrupt,
        };
        if meta.len() != ck.bytes {
            return ExactSet::Corrupt;
        }
        match meta.modified() {
            Ok(t) if fmt_mtime(t) == ck.mtime => {}
            _ => return ExactSet::Corrupt,
        }
    }
    // 2. No unlisted files anywhere in the subtree, no undeclared subdirs.
    let expected = expected_dirs(&record.artifacts);
    match scan_orphans(path, path, record, &expected) {
        Ok(()) => ExactSet::Ok,
        Err(e) => e,
    }
}

/// The set of directory rel-paths that *must* exist to hold the manifest's
/// nested files — every proper ancestor of every manifest key. A `chain_1/`
/// dir is legitimate because `chain_1/trace.tsv` is manifested; an empty or
/// stray `junk/` is not implied by any manifest entry and is therefore an
/// orphan (crash debris), preserving the strict "no undeclared subdir" gate
/// while supporting nesting.
fn expected_dirs(manifest: &BTreeMap<String, FileChecksum>) -> HashSet<String> {
    let mut dirs = HashSet::new();
    for key in manifest.keys() {
        let parts: Vec<&str> = key.split('/').collect();
        for i in 1..parts.len() {
            dirs.insert(parts[..i].join("/"));
        }
    }
    dirs
}

/// Recursive orphan scan rooted at the leaf `root`, currently visiting `dir`.
/// A file not in the manifest is an `Orphan`. A *declared child* subdir is a
/// boundary (recognized, not recursed). Any other subdir is an `Orphan`
/// unless it is an ancestor of a manifested file (in `expected`), in which
/// case it is part of the leaf and recursed into. Reserved files / `children`
/// keys are meaningful at the leaf root (`dir == root`) only.
fn scan_orphans(
    root: &Path,
    dir: &Path,
    record: &RunRecord,
    expected: &HashSet<String>,
) -> Result<(), ExactSet> {
    let entries = fs::read_dir(dir).map_err(|_| ExactSet::Corrupt)?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            if dir == root && record.children.contains_key(&name) {
                continue; // declared child boundary — its own leaf
            }
            // An undeclared dir is only legitimate if a manifested file lives
            // under it; an empty/stray dir is debris → orphan.
            if !expected.contains(&rel_key(root, &entry.path())) {
                return Err(ExactSet::Orphan);
            }
            scan_orphans(root, &entry.path(), record, expected)?;
        } else {
            if dir == root && is_reserved(&name) {
                continue;
            }
            if !record.artifacts.contains_key(&rel_key(root, &entry.path())) {
                return Err(ExactSet::Orphan);
            }
        }
    }
    Ok(())
}

/// Build the exact-set manifest of the leaf's OWN files by walking its whole
/// subtree (Mode B `finalize`): every file keyed by its `/`-joined relative
/// path, EXCEPT reserved files and the contents of declared `children`
/// subdirs (separate artifacts, manifested by their own commit).
fn build_manifest(
    root: &Path,
    children: &BTreeMap<String, Vec<ContentHash>>,
) -> Result<BTreeMap<String, FileChecksum>, CasError> {
    let mut manifest = BTreeMap::new();
    collect_own_files(root, root, children, &mut manifest)?;
    Ok(manifest)
}

fn collect_own_files(
    root: &Path,
    dir: &Path,
    children: &BTreeMap<String, Vec<ContentHash>>,
    out: &mut BTreeMap<String, FileChecksum>,
) -> Result<(), CasError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            if dir == root && children.contains_key(&name) {
                continue; // declared child boundary — its own leaf's bytes
            }
            collect_own_files(root, &entry.path(), children, out)?;
        } else {
            if dir == root && is_reserved(&name) {
                continue;
            }
            let bytes = fs::read(entry.path())?;
            out.insert(
                rel_key(root, &entry.path()),
                FileChecksum {
                    bytes: meta.len(),
                    mtime: fmt_mtime(meta.modified()?),
                    digest: ContentHash::digest_bytes(&bytes),
                },
            );
        }
    }
    Ok(())
}

/// `{seg}` → `{seg}~{hash16}` → `{seg}~{full64}`. The full-64 form *is* the
/// identity, so it cannot collide — the escalation always terminates.
fn disambiguation_candidates(path: &Path, run_id: &ContentHash) -> Vec<PathBuf> {
    let hex = run_id.to_hex();
    let parent = path.parent();
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mk = |suffix: Option<&str>| -> PathBuf {
        let seg = match suffix {
            Some(s) => format!("{name}~{s}"),
            None => name.clone(),
        };
        match parent {
            Some(p) => p.join(seg),
            None => PathBuf::from(seg),
        }
    };
    vec![mk(None), mk(Some(&hex[..16])), mk(Some(&hex))]
}

/// Remove every entry in `dir` except the `.lock` (Mode B reclaim).
fn clear_except_lock(dir: &Path) -> Result<(), CasError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name() == std::ffi::OsStr::new(".lock") {
            continue;
        }
        let p = entry.path();
        if entry.file_type()?.is_dir() {
            fs::remove_dir_all(&p)?;
        } else {
            fs::remove_file(&p)?;
        }
    }
    Ok(())
}

/// `"{secs}.{nanos:09}"` since the Unix epoch — deterministic, exactly
/// comparable.
fn fmt_mtime(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:09}", d.as_secs(), d.subsec_nanos())
}

fn fsync_dir(path: &Path) -> Result<(), CasError> {
    // On Unix a directory can be opened read-only and `sync_all`'d to make
    // its entries durable.
    File::open(path)?.sync_all()?;
    Ok(())
}

fn write_file_synced(path: &Path, bytes: &[u8]) -> Result<(), CasError> {
    let mut f = File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Write `run.json` directly (Mode A staging: the whole dir is renamed
/// atomically, so no tmp is needed).
fn write_record_direct(dir: &Path, record: &RunRecord) -> Result<(), CasError> {
    let json = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
    write_file_synced(&dir.join("run.json"), &json)
}

/// Write `run.json` via `run.json.tmp → rename` (Mode B: the leaf is visible
/// during streaming, so the single-file rename is the commit point).
fn write_record_atomic(dir: &Path, record: &RunRecord) -> Result<(), CasError> {
    let json = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
    let tmp = dir.join("run.json.tmp");
    write_file_synced(&tmp, &json)?;
    fs::rename(&tmp, dir.join("run.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests;
