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
//! ## Stale-lock reclaim
//!
//! A fit killed mid-run (Ctrl-C / SIGPIPE / OOM / crash) never runs
//! `finalize`, so its `.lock` + `Running` `run.json` linger. A re-run reclaims
//! such a leaf: the claim liveness-checks the lock's recorded PID
//! (`kill(pid, 0)` on unix). A **dead** holder is a stale lock — removed and
//! re-claimed, clearing the dead run's orphan contents. A **live** holder is a
//! genuine concurrent claim and still returns `FitInProgress`. An
//! unreadable/zero PID we can't verify is treated conservatively as live. On
//! non-unix the PID can't be checked, so a held lock is always `FitInProgress`
//! (no auto-reclaim). A stale `Running` leaf with **no** `.lock` is likewise
//! reclaimed (the `O_EXCL` create simply succeeds).
//!
//! Reclaim is **serialized through a second `.reclaim` lock** so it stays
//! exclusive under concurrent re-launch (e.g. a cluster re-submit of a killed
//! fit). Only the `.reclaim` holder may remove `.lock`, and it re-confirms the
//! holder is dead while holding `.reclaim`; a bare `create_new(.lock)` never
//! removes anything. While the dead `.lock` exists, every other claimant's
//! `create_new(.lock)` fails and funnels into the reclaim path, where
//! `.reclaim` blocks it — so two processes can never both delete `.lock` and
//! both enter the critical section. A claimant never touches another's live
//! `.reclaim`/`.lock.new` serializers (clearing them would re-open the
//! double-reclaim race), so a serializer left behind by a process killed within
//! the few-syscall reclaim window is *not* cleared on the next acquire: it makes
//! the leaf refuse a dead-lock reclaim until removed. That is safe, never
//! corrupting — the serializers are excluded from the manifest + orphan scan and
//! the window holds no fit work — and the proper home for sweeping such debris
//! is the store-open lifecycle sweep, not the claim path.

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
    /// The Mode B `.lock` is held by a live process (a genuine concurrent
    /// fit), or by a PID we can't verify, or a reclaim is already in flight. A
    /// lock held by a dead PID is reclaimed instead (see the stale-lock reclaim
    /// note on this module).
    #[error("fit in progress at {path} (pid {pid})")]
    FitInProgress { path: String, pid: u32 },
    /// A `claim_streaming` found a `Completed` same-identity leaf — the caller
    /// should have taken the cache hit from `lookup` instead of claiming.
    #[error("artifact already completed at {path}")]
    AlreadyCompleted { path: String },
    /// Defense-in-depth: after winning the lock, the leaf's `run.json` was
    /// already `Completed`. A legitimate reclaim only ever clears a
    /// crashed/incomplete leaf; a `Completed` record at clear-time means a
    /// concurrent claimant finalized this leaf out from under us (a reclaim
    /// race residual). Refuse loudly rather than blind-wipe a finished result.
    #[error("refusing to clear a Completed leaf at {path} (concurrent finalize)")]
    ReclaimRaceCompleted { path: String },
    /// Same identity, different bytes: a commit found a `Completed` incumbent
    /// whose stored file disagrees with what was just staged. Runs are
    /// seeded-deterministic, so this is an identity bug — an input that
    /// changes the output is missing from the `run_id` — or nondeterminism;
    /// either way the two results must not share a key, and silently
    /// discarding the staged bytes (the pre-S1 behavior) served the
    /// incumbent as if it were this run's result. The staged directory is
    /// preserved under `.quarantine/` as evidence. Proposal:
    /// docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md §S1.
    #[error(
        "divergent recompute at {path}: '{file}' staged {ours} vs stored {theirs} \
         under one run_id — an input is missing from the identity, or the run \
         is nondeterministic. Staged bytes preserved in .quarantine/."
    )]
    DivergentRecompute { path: String, file: String, ours: String, theirs: String },
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
    /// The overwrite door: displace a `Completed` leaf so the next claim or
    /// commit at this identity starts clean.
    ///
    /// The store had no such operation, which is why `--force` meant four
    /// different things at five call sites and "overwrite" at none of them:
    /// batch recomputed and then had its bytes discarded by the
    /// already-completed path, fit and survey died with
    /// `AlreadyCompleted`, and simulate's flag was inert. Forcing routes
    /// through the same collision-aware machinery as everything else — it
    /// only ever displaces a leaf whose identity MATCHES (a different
    /// identity at this path is somebody else's artifact), and it
    /// **quarantines** rather than deletes, so a forced recompute never
    /// destroys the result it replaces.
    ///
    /// A no-op when the leaf is absent, incomplete, or holds a different
    /// identity: those already recompute without help.
    pub fn displace_completed(&self, path: &Path, expected: &LeafIdentity) -> Result<(), CasError> {
        if matches!(self.lookup(path, expected), Lookup::Hit(_)) {
            // Never displace a leaf a live process is holding.
            if held_by_live_lock(path) {
                let pid = read_lock_pid(&path.join(".lock")).unwrap_or(0);
                return Err(CasError::FitInProgress { path: path.display().to_string(), pid });
            }
            self.quarantine(path)?;
        }
        Ok(())
    }

    /// Add one artifact to a leaf that is already `Completed`.
    ///
    /// The store had no way for a finished leaf to gain a file, and the gap
    /// was load-bearing: `simulate --event-log` stages `event_log.tsv` into a
    /// leaf that may already exist, and the whole staged set was then
    /// discarded as an already-completed no-op, so the log was silently lost.
    /// The in-code note said `--force` was the workaround; it was not (that
    /// path re-commits and hits the same discard). This is the operation both
    /// wanted.
    ///
    /// Contract:
    /// - the leaf must be `Completed` AND carry `expected`'s identity — a
    ///   different identity at this path is someone else's artifact;
    /// - a live `.lock` holder blocks (someone is claiming/reclaiming);
    /// - re-adding the SAME bytes is an idempotent no-op, so a rerun that
    ///   records the log again is harmless;
    /// - re-adding DIFFERENT bytes under one name is
    ///   [`CasError::DivergentRecompute`], for the reason the commit path
    ///   gives: same key, same bytes, or it is an identity bug.
    ///
    /// Identity is untouched — `artifacts` is recorded, not hashed — so a leaf
    /// that gains a file keeps its `run_id`.
    ///
    /// Crash-safety: the file is written and fsync'd BEFORE the record names
    /// it. A crash in that window leaves an unmanifested file, which the
    /// exact-set scan reports as `Stale(OrphanFiles)` and the next run
    /// recomputes — the safe direction, never a manifest pointing at bytes
    /// that were never flushed.
    pub fn augment(
        &self,
        path: &Path,
        expected: &LeafIdentity,
        name: &str,
        bytes: &[u8],
    ) -> Result<(), CasError> {
        // ADVISORY pre-check, deliberately outside the lock: it only decides
        // whether there is anything here worth locking for. Every decision
        // that matters — idempotent re-add, divergence, the manifest we write
        // back — is re-made under the lock below, against a fresh read.
        //
        // This ordering is load-bearing. `augment` is the store's only
        // read-modify-write of `run.json` (`commit_atomic` gets its atomicity
        // from staging + rename instead), so checking a pre-lock snapshot and
        // writing it back after locking loses concurrent work two ways: two
        // augments of the SAME name with different bytes both pass the check
        // and the second silently overwrites the first (defeating the very
        // guarantee `DivergentRecompute` exists to make), and two augments of
        // DIFFERENT names each write back a snapshot missing the other's
        // entry, orphaning a file and staleing a completed leaf. `batch`
        // augments exactly such a pair (`event_log.tsv`, `reactive_log.tsv`).
        match self.lookup(path, expected) {
            Lookup::Hit(_) => {}
            Lookup::Collision(_) => {
                return Err(CasError::AlreadyCompleted { path: path.display().to_string() })
            }
            _ => {
                // Not a completed leaf of this identity: nothing to augment.
                // The caller's next run will write it from scratch.
                return Ok(());
            }
        }

        // TEST-ONLY: fire at the instant a caller has read the leaf but not
        // yet locked it — the window in which the pre-fix code made its
        // idempotency/divergence decision. A test drives a full competing
        // augment through here and asserts this caller, on resume, sees the
        // competitor's write under the lock instead of clobbering it.
        #[cfg(test)]
        augment_gap_hook(path);

        // Take the leaf's lock. A `.lock` on a COMPLETED leaf is either a
        // concurrent augment (refuse) or debris from a crashed one (take
        // over) — the same question every other lock consumer in this file
        // asks, so it routes through the same answer. A bare
        // `AlreadyExists -> FitInProgress` here would report "fit in progress"
        // under a dead pid and wedge the leaf until `--force`, which is the
        // failure `reclaim_or_refuse`'s own comment warns about.
        let lock = path.join(".lock");
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(mut f) => {
                write!(f, "{}", std::process::id())?;
                f.sync_all()?;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                reclaim_or_refuse(path, &lock)?;
            }
            Err(e) => return Err(e.into()),
        }

        let outcome = (|| -> Result<(), CasError> {
            // Re-read UNDER the lock: the snapshot above may be stale.
            let mut record = match self.lookup(path, expected) {
                Lookup::Hit(r) => r,
                Lookup::Collision(_) => {
                    return Err(CasError::AlreadyCompleted { path: path.display().to_string() })
                }
                // The leaf went away (or was reclaimed) between the two reads.
                _ => return Ok(()),
            };

            let digest = ContentHash::digest_bytes(bytes);
            if let Some(existing) = record.artifacts.get(name) {
                if existing.digest == digest {
                    return Ok(());
                }
                // Preserve the rejected bytes, as `DivergentRecompute`
                // promises: the one signal the store emits for a suspected
                // identity bug must not send the reader to an empty directory.
                let kept = self.quarantine_bytes(path, name, bytes, &digest)?;
                return Err(CasError::DivergentRecompute {
                    path: path.display().to_string(),
                    file: name.to_string(),
                    ours: format!("{} (kept at {})", digest.to_hex(), kept.display()),
                    theirs: existing.digest.to_hex(),
                });
            }

            let fp = path.join(name);
            if let Some(parent) = fp.parent() {
                fs::create_dir_all(parent)?;
            }
            write_file_synced(&fp, bytes)?;
            let meta = fs::metadata(&fp)?;
            record.artifacts.insert(
                name.to_string(),
                FileChecksum {
                    bytes: bytes.len() as u64,
                    mtime: fmt_mtime(meta.modified()?),
                    digest,
                },
            );
            write_record_atomic(path, &record)?;
            fsync_dir(path)?;
            Ok(())
        })();

        fs::remove_file(&lock).ok();
        outcome
    }

    /// Preserve bytes an `augment` refused, under `.quarantine/`, and return
    /// where they landed. The commit path quarantines a whole staging dir;
    /// this is the single-file analogue, so both routes to
    /// [`CasError::DivergentRecompute`] leave the evidence its message names.
    fn quarantine_bytes(
        &self,
        leaf: &Path,
        name: &str,
        bytes: &[u8],
        digest: &ContentHash,
    ) -> Result<PathBuf, CasError> {
        let q = self.quarantine_root();
        fs::create_dir_all(&q)?;
        let leaf_name = leaf
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "leaf".into());
        // Artifact names may nest (`chain_1/trace.tsv`); flatten so the
        // evidence is one file, and tag it with the digest so two rejected
        // versions of one name cannot overwrite each other.
        let flat = name.replace(['/', '\\'], "_");
        let dest = q.join(format!("{}.{}.{}", leaf_name, flat, &digest.to_hex()[..8]));
        write_file_synced(&dest, bytes)?;
        Ok(dest)
    }

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
        self.rename_staged_into_place(&staging, path, &id, &record.artifacts)
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
        staged_manifest: &BTreeMap<String, FileChecksum>,
    ) -> Result<PathBuf, CasError> {
        // The loop converges fast: a competing same-identity commit resolves
        // to AlreadyCompleted, a different identity disambiguates to its own
        // candidate. The cap is a defensive backstop, never reached in
        // practice.
        const MAX_ATTEMPTS: u32 = 64;
        for _ in 0..MAX_ATTEMPTS {
            match self.resolve_claim_dir(path, id)? {
                ClaimOutcome::AlreadyCompleted(dest) => {
                    // S1 divergence check (proposal 2026-08-23-run-identity-
                    // and-store-contract): a same-identity incumbent is a
                    // benign dedup ONLY if its bytes match what we staged.
                    // "Same key ⟺ same bytes" is the store's whole claim;
                    // this is the one moment both sides are in hand. A shared
                    // file with a differing digest means an identity bug or
                    // nondeterminism — quarantine the staged evidence and
                    // fail loudly instead of silently serving the incumbent.
                    let incumbent = match read_record(&dest) {
                        ReadResult::Ok(r) => r,
                        // The incumbent vanished or tore between resolve and
                        // read (e.g. a concurrent reclaim): re-resolve.
                        ReadResult::Absent | ReadResult::Unparseable => continue,
                    };
                    for (name, ours) in staged_manifest {
                        if let Some(theirs) = incumbent.artifacts.get(name) {
                            if ours.digest != theirs.digest {
                                self.quarantine(staging)?;
                                return Err(CasError::DivergentRecompute {
                                    path: dest.display().to_string(),
                                    file: name.clone(),
                                    ours: ours.digest.to_hex(),
                                    theirs: theirs.digest.to_hex(),
                                });
                            }
                        }
                    }
                    // All shared files agree. Staged-only extras (the "leaf
                    // gains an artifact" case, e.g. --event-log over a
                    // pre-existing leaf) are dropped here until the S4
                    // augment door lands; surfacing that to the caller is
                    // S2's WriteVerdict (this crate has no logging channel,
                    // so the proposal's "report" deliberately waits for it).
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

        // The O_EXCL `.lock` is the authoritative race guard: whoever creates
        // it holds the leaf. If it already exists, `reclaim_or_refuse` decides
        // whether the holder is a dead crash (reclaim, serialized through
        // `.reclaim`) or a live concurrent claim (refuse). A bare claimant
        // never removes a lock, so concurrent reclaimers can't double-enter.
        let lock = dir.join(".lock");
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(mut f) => {
                write!(f, "{}", std::process::id())?;
                f.sync_all()?;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                reclaim_or_refuse(&dir, &lock)?;
            }
            Err(e) => return Err(e.into()),
        }

        // We now hold the exclusive lock. Defense-in-depth: a legitimate
        // reclaim only ever clears a crashed/incomplete leaf (`resolve_claim_dir`
        // returns `Reclaim`/`Fresh` only for non-`Completed` states). If the
        // `run.json` is `Completed` here, a concurrent claimant finalized this
        // leaf out from under us — refuse loudly rather than blind-wipe a
        // finished result. (Fresh claims never see a `Completed` same-identity
        // run.json: that would have surfaced as `AlreadyCompleted` above.)
        if let ReadResult::Ok(r) = read_record(&dir) {
            if r.status == RunStatus::Completed {
                return Err(CasError::ReclaimRaceCompleted { path: dir.display().to_string() });
            }
        }

        // Clear any stale orphan contents (a prior crashed run's chain files +
        // stale run.json), preserving only our fresh `.lock`.
        clear_except_lock(&dir)?;

        // TEST-ONLY: fire at the instant the leaf holds our live `.lock` but no
        // `run.json` (just cleared, not yet rewritten). A concurrent claimant
        // reading the leaf here sees `Lookup::Miss`; the gate must route it
        // through the live `.lock` (→ `FitInProgress`), not quarantine the dir
        // out from under us. Inert in non-test builds.
        #[cfg(test)]
        clear_gap_hook(&dir);

        let mut rec = running;
        rec.status = RunStatus::Running;
        rec.artifacts = BTreeMap::new();
        write_record_atomic(&dir, &rec)?;
        fsync_dir(&dir)?;
        Ok(StreamClaim { dir, finalized: false })
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
                    // Never quarantine a leaf a live process is holding: route
                    // it to the `.lock` gate (`reclaim_or_refuse`) instead.
                    if held_by_live_lock(&cand) {
                        return Ok(ClaimOutcome::Fresh(cand));
                    }
                    self.quarantine(&cand)?;
                    return Ok(ClaimOutcome::Fresh(cand));
                }
                // A different full identity shares this short-hash segment:
                // advance to the disambiguated candidate, never touch it.
                Lookup::Collision(_) => continue,
                // Dir exists with no run.json: an orphaned partial → quarantine.
                // But a live `.lock` holder transiently shows no run.json while
                // it runs `clear_except_lock` then rewrites it; quarantining
                // there would rip the dir out from under the active holder and
                // race its writes to a `NotFound`. A live lock ⇒ not orphan
                // debris: route to the `.lock` gate, which refuses with
                // `FitInProgress` (live) or reclaims (dead) — never quarantines.
                Lookup::Miss => {
                    if held_by_live_lock(&cand) {
                        return Ok(ClaimOutcome::Fresh(cand));
                    }
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
    /// Set by [`finalize`](Self::finalize) once the leaf is `Completed`. While
    /// false, [`Drop`] treats the claim as abandoned and marks the leaf
    /// `Failed` — see the impl below.
    finalized: bool,
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
    pub fn finalize(mut self, mut record: RunRecord) -> Result<PathBuf, CasError> {
        // Walk the whole streamed subtree (chains nest under `chain_N/`),
        // stopping at the declared children (`trajectories/`, `dt_check/`,
        // `obs/`) — those are separate artifacts, manifested by their own
        // commit, never folded into this leaf's exact set.
        record.artifacts = build_manifest(&self.dir, &record.children)?;
        record.status = RunStatus::Completed;
        write_record_atomic(&self.dir, &record)?;
        fsync_dir(&self.dir)?;
        fs::remove_file(self.dir.join(".lock")).ok();
        // Only now is the leaf durably Completed; `Drop` must not touch it.
        // Set AFTER the commit point so a failure above still drops as Failed.
        self.finalized = true;
        Ok(self.dir.clone())
    }
}

/// An abandoned claim marks its leaf `Failed` and releases the lock.
///
/// Before this, a claim dropped without `finalize` — an error return, a `?`,
/// a panic unwind — left the leaf `Running` with a live `.lock`, and
/// `RunStatus::Failed` was declared but written nowhere: a cleanly-failed run
/// and a `kill -9` were indistinguishable on disk, and the next same-identity
/// claimant had to wait for PID-liveness reclaim to clear it.
///
/// Best-effort by necessity (`Drop` cannot report): every step is `.ok()`'d.
/// PID reclaim remains the backstop for the paths `Drop` cannot see — a
/// `kill -9`, and `process::exit` (which runs no destructors).
impl Drop for StreamClaim {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        if let ReadResult::Ok(mut record) = read_record(&self.dir) {
            record.status = RunStatus::Failed;
            write_record_atomic(&self.dir, &record).ok();
        }
        fs::remove_file(self.dir.join(".lock")).ok();
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
/// its tmp, and the Mode B locks (`.lock`, its atomic-rename temp `.lock.new`,
/// and the `.reclaim` serializer).
fn is_reserved(name: &str) -> bool {
    matches!(name, "run.json" | "run.json.tmp" | ".lock" | ".lock.new" | ".reclaim")
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
/// The per-file `digest` is recorded in `run.json` for integrity tooling
/// (`camdl verify`); no read path recomputes it today.
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
        // Skip the reserved lock siblings BEFORE stat'ing them. A concurrent
        // claimant churns `.reclaim`/`.lock.new` in this same leaf; if such a
        // transient is enumerated by `read_dir` then removed before our
        // `metadata()`, the `?` would bubble `Io(NotFound)` out of finalize.
        if dir == root && is_reserved(&name) {
            continue;
        }
        let meta = entry.metadata()?;
        if meta.is_dir() {
            if dir == root && children.contains_key(&name) {
                continue; // declared child boundary — its own leaf's bytes
            }
            collect_own_files(root, &entry.path(), children, out)?;
        } else {
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
/// Whether process `pid` is currently alive. Unix: `kill(pid, 0)` succeeds
/// (exists, ours) or fails with `EPERM` (exists, not ours) → alive; only
/// `ESRCH` (no such process) → dead. Non-unix: conservatively `true` — we
/// can't check, so we never reclaim a lock we can't prove is stale.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    // SAFETY: signal 0 delivers nothing; it only probes existence/permission.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

/// The PID recorded in a `.lock`, if present and parseable.
fn read_lock_pid(lock: &Path) -> Option<u32> {
    fs::read_to_string(lock).ok().and_then(|s| s.trim().parse().ok())
}

/// Whether `dir` is currently held by a live `.lock` holder — i.e. some
/// process is actively claiming/reclaiming this leaf right now. Used to keep
/// `resolve_claim_dir` from quarantining a leaf out from under its legitimate
/// holder: while the holder runs `clear_except_lock` it transiently removes
/// `run.json`, and a concurrent claimant that read the leaf in that window
/// would otherwise see `Lookup::Miss` and `rename` the whole (actively
/// written) dir away — racing the holder's own writes to a `NotFound` error.
/// A live `.lock` means "not orphan debris, route to the lock gate instead".
fn held_by_live_lock(dir: &Path) -> bool {
    matches!(read_lock_pid(&dir.join(".lock")), Some(p) if p != 0 && pid_is_alive(p))
}

/// A `.lock` already exists at claim time. Reclaim it iff its holder PID is
/// **provably dead**, serializing the reclaim through a `.reclaim` lock so
/// concurrent reclaimers can never both remove `.lock` and both enter the
/// critical section. A live/unverifiable holder — or a reclaim already in
/// flight — refuses with `FitInProgress`. On success a freshly created `.lock`
/// records our PID and the caller proceeds to clear orphans + write `Running`.
fn reclaim_or_refuse(dir: &Path, lock: &Path) -> Result<(), CasError> {
    let fail = |pid: u32| CasError::FitInProgress { path: dir.display().to_string(), pid };
    let holder = read_lock_pid(lock);
    // Only reclaim what we can PROVE is stale. An unreadable/zero PID we can't
    // verify, or a live one, is a genuine concurrent claim.
    if !matches!(holder, Some(p) if p != 0 && !pid_is_alive(p)) {
        return Err(fail(holder.unwrap_or(0)));
    }
    // TEST-ONLY: fire after the dead-check, before acquiring `.reclaim`. A test
    // parks the losing reclaimer here while a concurrent winner completes a full
    // takeover (its live PID over `.lock`) and releases `.reclaim`; on resume
    // this thread re-acquires the recycled `.reclaim`, and the re-confirm below
    // must catch the now-live lock and refuse rather than double-reclaim. Inert
    // in non-test builds.
    #[cfg(test)]
    reclaim_preacquire_hook(lock);
    // Serialize the reclaim: only the `.reclaim` holder may remove `.lock`.
    // Contention here means another process is already reclaiming/claiming.
    let reclaim = dir.join(".reclaim");
    match OpenOptions::new().write(true).create_new(true).open(&reclaim) {
        Ok(mut rf) => {
            let _ = write!(rf, "{}", std::process::id());
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            // A live serializer means a genuine concurrent reclaim — refuse.
            // But a process that died holding `.reclaim` strands it, and an
            // unconditional refusal here wedges the leaf permanently: every
            // future reclaim sees AlreadyExists and gives up, so the leaf can
            // never be recomputed without manual `rm`. The module doc deferred
            // this to a store-open sweep that was never written; clearing it
            // at the point of contention needs no store-wide walk.
            //
            // `.reclaim` records its creator's pid, so the same liveness test
            // that governs `.lock` governs it. If two processes both find it
            // stranded, both remove and retry: one wins `create_new`, the
            // other sees a LIVE holder and refuses correctly.
            let stranded = read_lock_pid(&reclaim).is_some_and(|p| !pid_is_alive(p));
            if !stranded {
                return Err(fail(holder.unwrap_or(0)));
            }
            let _ = fs::remove_file(&reclaim);
            match OpenOptions::new().write(true).create_new(true).open(&reclaim) {
                Ok(mut rf) => {
                    let _ = write!(rf, "{}", std::process::id());
                }
                // Lost the retry to another claimant — it holds the gate now.
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    return Err(fail(holder.unwrap_or(0)));
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(e) => return Err(e.into()),
    }
    // Re-confirm the holder is STILL the dead PID we saw, now that we hold
    // `.reclaim`. Between the dead-check above and acquiring `.reclaim`, a
    // concurrent reclaimer can complete a full takeover (atomic-rename its own
    // live PID over `.lock`) AND release `.reclaim` — letting us re-acquire the
    // now-free `.reclaim` and take over a lock that is no longer dead, putting
    // two claimants in the critical section. The `.reclaim` O_EXCL gate only
    // serializes the takeover, not the run, and the PID we validated is stale
    // by the time we own the gate; so the liveness check must be redone here.
    let now = read_lock_pid(lock);
    if now != holder {
        let _ = fs::remove_file(&reclaim);
        return Err(fail(now.unwrap_or(0)));
    }
    // Take over the dead `.lock` WITHOUT ever leaving it absent. A bare
    // claimant's `create_new(.lock)` must always fail (file present) so it
    // funnels into this gate, where `.reclaim` blocks it — never a window in
    // which `.lock` is gone and a bare `create_new` could succeed. So we write
    // the live PID to a temp sibling, fsync it, then atomically `rename` it
    // over `.lock`. `rename(2)` is atomic within a filesystem: `.lock` goes
    // dead-PID → live-PID in one step, never absent. We hold `.reclaim`
    // throughout; release it on every path.
    let lock_new = dir.join(".lock.new");
    // TEST-ONLY: fire the gap hook at the point the old code left `.lock`
    // absent. Here `.lock` still holds the dead PID, so a concurrent bare
    // claimant's create_new(.lock) fails → routes to reclaim → refuses. Inert
    // in non-test builds.
    #[cfg(test)]
    reclaim_gap_hook(lock);
    let outcome = (|| -> Result<(), CasError> {
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&lock_new)?;
        write!(f, "{}", std::process::id())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&lock_new, lock)?;
        Ok(())
    })();
    if outcome.is_err() {
        let _ = fs::remove_file(&lock_new);
    }
    let _ = fs::remove_file(&reclaim);
    outcome
}

fn clear_except_lock(dir: &Path) -> Result<(), CasError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        // Never touch the lock serializers: a concurrent claimant legitimately
        // churns `.reclaim`/`.lock.new` in this leaf, and `.lock` is ours.
        // These are not crashed-run debris for us to clear.
        if is_reserved(&name.to_string_lossy()) {
            continue;
        }
        let p = entry.path();
        // Tolerate a sibling that vanished between the `read_dir` snapshot and
        // here (a concurrent claimant's transient): it is already gone, which
        // is the outcome we wanted — not our error.
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) if e.kind() == ErrorKind::NotFound => continue,
            Err(e) => return Err(e.into()),
        };
        let r = if ft.is_dir() {
            fs::remove_dir_all(&p)
        } else {
            fs::remove_file(&p)
        };
        if let Err(e) = r {
            if e.kind() != ErrorKind::NotFound {
                return Err(e.into());
            }
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

/// TEST-ONLY hook fired inside [`FsCasStore::augment`] after its advisory
/// pre-read and before it takes the leaf lock — the window where the pre-fix
/// code decided idempotency/divergence on a snapshot it later wrote back. A
/// test drives a competing augment through this instant and asserts the
/// resuming caller re-reads under the lock rather than clobbering. `None` for
/// every test that does not opt in.
#[cfg(test)]
type AugmentGapHook = std::sync::Arc<dyn Fn(&Path) + Send + Sync>;

#[cfg(test)]
static AUGMENT_GAP_HOOK: std::sync::Mutex<Option<AugmentGapHook>> = std::sync::Mutex::new(None);

/// TEST-ONLY: serializes the tests that install [`AUGMENT_GAP_HOOK`], a
/// process-global, so it never fires inside another test's augment.
#[cfg(test)]
static AUGMENT_HOOK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn augment_gap_hook(leaf: &Path) {
    // Cloned out and invoked with the mutex RELEASED: the driven competitor
    // re-enters `augment` and would self-deadlock on a held guard.
    let hook = AUGMENT_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(h) = hook {
        h(leaf);
    }
}

/// TEST-ONLY hook fired inside `reclaim_or_refuse`, at the instant just before
/// the dead `.lock` is taken over by the atomic rename — the point at which the
/// pre-fix code left `.lock` observably absent. A test installs a closure here
/// to drive a concurrent bare claimant into that instant and assert it is now
/// refused (the TOCTOU proof). `None` for every test that does not opt in, so
/// the reclaim path is unchanged for the rest of the suite.
#[cfg(test)]
type ReclaimGapHook = Box<dyn Fn(&Path) + Send>;

#[cfg(test)]
static RECLAIM_GAP_HOOK: std::sync::Mutex<Option<ReclaimGapHook>> = std::sync::Mutex::new(None);

/// TEST-ONLY: serializes the tests that exercise `reclaim_or_refuse`, so an
/// installed `RECLAIM_GAP_HOOK` (a process-global) never fires in another
/// concurrently-running test's reclaim.
#[cfg(test)]
static RECLAIM_HOOK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn reclaim_gap_hook(lock: &Path) {
    let guard = RECLAIM_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = guard.as_ref() {
        h(lock);
    }
}

/// TEST-ONLY hook fired inside `reclaim_or_refuse` AFTER the dead-PID check but
/// BEFORE `.reclaim` is acquired — the instant a losing reclaimer has committed
/// to "the holder is dead" on a now-stale read. A test drives a concurrent
/// winner through a full takeover here and asserts the loser, on resume, is
/// refused (the `.reclaim`-recycle / double-reclaim proof). Held behind an
/// `Arc` and invoked with the mutex released (the driven winner re-enters this
/// hook, so holding the lock across the call would self-deadlock); a fire-once
/// guard in the test keeps it from firing on the winner's own reclaim.
#[cfg(test)]
type ReclaimPreacquireHook = std::sync::Arc<dyn Fn(&Path) + Send + Sync>;

#[cfg(test)]
static RECLAIM_PREACQUIRE_HOOK: std::sync::Mutex<Option<ReclaimPreacquireHook>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
fn reclaim_preacquire_hook(lock: &Path) {
    let hook = RECLAIM_PREACQUIRE_HOOK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(h) = hook {
        h(lock);
    }
}

/// TEST-ONLY hook fired inside `claim_streaming` at the instant the winning
/// claimant holds the `.lock` (its own live PID) but has just `clear_except_lock`'d
/// the leaf — so `run.json` is transiently absent. A test installs a closure to
/// drive a concurrent claimant through `claim_streaming` at that instant and
/// assert it does NOT quarantine the actively-held leaf (the `Lookup::Miss`
/// false-orphan race). `None` for every test that does not opt in.
///
/// Held behind an `Arc` (not a `Box` like the reclaim hook) and invoked with
/// the mutex *released*: the closure drives a concurrent claimant whose own
/// `claim_streaming` may re-enter `clear_gap_hook`, so holding the mutex across
/// the call would self-deadlock. Cloning the `Arc` under the lock and calling
/// through the clone keeps re-entry lock-free.
#[cfg(test)]
type ClearGapHook = std::sync::Arc<dyn Fn(&Path) + Send + Sync>;

#[cfg(test)]
static CLEAR_GAP_HOOK: std::sync::Mutex<Option<ClearGapHook>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn clear_gap_hook(dir: &Path) {
    let hook = CLEAR_GAP_HOOK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(h) = hook {
        h(dir);
    }
}

#[cfg(test)]
mod tests;
