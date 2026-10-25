//! A rebuildable derived index over the content-addressed store, caching
//! `run_id → (rel_path, kind, label, status, created_at)` so `show`/`cat`
//! prefix resolution does not re-walk the whole `results/` tree on every call.
//!
//! **`run.json` is the operational source of truth; the index is never
//! authoritative.** Two invariants make that operational rather than
//! aspirational:
//!
//! 1. **Miss → full walk → repair.** A prefix lookup the index cannot satisfy
//!    falls back to [`cas_read::walk_records`] (today's behaviour) and then
//!    refreshes the index from that walk. An out-of-band-added leaf (created by
//!    a concurrent writer, or by anything that did not update the index) is
//!    still found — never reported "no match" merely because the index lacks
//!    it.
//! 2. **Stale entry → drop + re-walk.** Every index hit is *verified* against
//!    the live tree: the entry's `run.json` is re-read and its `run_id`
//!    re-checked. An entry whose `run.json` is gone (the leaf was `rm -rf`'d)
//!    or now holds a different identity is dropped, and the lookup falls
//!    through to the walk. The index never resolves a `run_id` to a dead or
//!    wrong path.
//!
//! Writes use the same atomic tmp + rename + fsync ordering as the store's
//! `run.json` (see `runid::store`), so a concurrent `simulate`/`fit` process can
//! neither observe nor produce a torn `index.json`. A malformed or absent
//! `index.json` is a clean miss (fall back to the walk), never a panic.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use runid::{ArtifactKind, RunRecord};
use serde::{Deserialize, Serialize};

use crate::cas_read::{self, Leaf};

/// The `index.json` schema version. A clean break bumps this; a mismatched (or
/// absent) version is a clean miss, never a deserialization error surfaced to
/// the user.
pub const INDEX_VERSION: u16 = 1;

/// The on-disk index filename, at the store root (`<root>/index.json`).
const INDEX_FILE: &str = "index.json";
const INDEX_TMP: &str = "index.json.tmp";

/// One cached leaf: enough to address it (`run_id` + `rel_path`) and to render
/// a `list` row without re-reading `run.json` (`kind`, `label`, `status`,
/// `created_at`). The `rel_path` is relative to the store root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Full 64-char hex `run_id` — the canonical address.
    pub run_id: String,
    /// Path to the leaf directory, relative to the store root.
    pub rel_path: String,
    pub kind: ArtifactKind,
    /// The leaf's display label (the last factored level's label, the
    /// human-facing name), recorded for fast `list` rendering.
    pub label: String,
    pub status: runid::RunStatus,
    /// `provenance.created_at` (ISO-8601), if recorded.
    pub created_at: Option<String>,
}

/// The derived index: a versioned list of leaf entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CasIndex {
    pub version: u16,
    pub entries: Vec<IndexEntry>,
}

impl CasIndex {
    fn new(entries: Vec<IndexEntry>) -> Self {
        Self { version: INDEX_VERSION, entries }
    }

    /// Load `<root>/index.json`. A missing file, an unparseable file, or a
    /// version mismatch all yield `None` (a clean miss) — never a panic.
    pub fn load(root: &Path) -> Option<Self> {
        let bytes = fs::read(root.join(INDEX_FILE)).ok()?;
        let idx: CasIndex = serde_json::from_slice(&bytes).ok()?;
        if idx.version != INDEX_VERSION {
            return None;
        }
        Some(idx)
    }
}

/// Build an [`IndexEntry`] from a discovered leaf record + its directory.
fn entry_from(root: &Path, dir: &Path, rec: &RunRecord) -> IndexEntry {
    let rel_path = dir
        .strip_prefix(root)
        .unwrap_or(dir)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/");
    // The display label is the last factored level's label (the most specific
    // segment, e.g. `seed_42`); empty if a record somehow has no levels.
    let label = rec.levels.last().map(|l| l.label.clone()).unwrap_or_default();
    IndexEntry {
        run_id: rec.run_id.to_hex(),
        rel_path,
        kind: rec.kind,
        label,
        status: rec.status,
        created_at: rec.provenance.created_at.clone(),
    }
}

/// Rebuild the index from a fresh full walk of every `run.json` under `root`,
/// then write it atomically. Returns the number of entries indexed.
///
/// This is the `camdl reindex` body and the repair path: it drops any entry
/// whose `run.json` is no longer present (they simply are not re-discovered by
/// the walk) and adds every leaf found.
pub fn rebuild(root: &Path) -> std::io::Result<usize> {
    let entries = walk_all_entries(root);
    let n = entries.len();
    write_atomic(root, &CasIndex::new(entries))?;
    Ok(n)
}

/// Walk the whole tree and project every discovered `RunRecord` to an
/// [`IndexEntry`]. The single source of discovery is `cas_read::walk_records`,
/// so the index can never claim a leaf the live walk would not.
fn walk_all_entries(root: &Path) -> Vec<IndexEntry> {
    cas_read::walk_records(root)
        .into_iter()
        .map(|(dir, rec)| entry_from(root, &dir, &rec))
        .collect()
}

/// Index-accelerated prefix resolution for one kind.
///
/// Returns `Some(leaves)` only when the index produced a verified, non-empty
/// match set (the fast path); the caller uses it directly. Returns `None` on
/// any of: no index, no indexed candidate of this kind matches the prefix, or
/// every candidate failed live verification (stale). `None` means "fall back
/// to the full walk" — which both finds out-of-band leaves (invariant 1) and
/// is the only path that can prove a true "no match".
///
/// Every returned [`Leaf`] is built by re-reading its live `run.json` and
/// re-checking that the on-disk `run_id` still matches the indexed one, so a
/// stale or repointed entry is dropped here (invariant 2) — the index can
/// never resolve a `run_id` to a dead or wrong path.
pub fn resolve_prefix(root: &Path, kind: ArtifactKind, prefix: &str) -> Option<Vec<Leaf>> {
    // Ambiguity safety: the index fast path is sound ONLY for an exact,
    // full-length `run_id` (64 hex). A full id is unique, so a verified index
    // hit is provably the complete match set and `show`/`cat` cannot
    // mis-resolve. For a SHORTER prefix the index may be missing an
    // out-of-band leaf that *also* matches (writers don't yet maintain the
    // index, so freshly-written leaves are routinely un-indexed); trusting a
    // non-empty indexed result would silently under-report ambiguity —
    // resolving `show <prefix>` to one of several matches instead of erroring.
    // So short prefixes always fall through (`None`) to the authoritative
    // walk, which enumerates every match and detects ambiguity correctly.
    // (Writer-side index maintenance would make the index complete and let
    // short prefixes use the fast path too; until then, exact-id only.)
    const RUN_ID_HEX_LEN: usize = 64;
    if prefix.len() != RUN_ID_HEX_LEN {
        return None;
    }
    let idx = CasIndex::load(root)?;
    let mut out = Vec::new();
    for entry in &idx.entries {
        if entry.kind != kind || !entry.run_id.starts_with(prefix) {
            continue;
        }
        // Verify against the live tree: the entry's leaf must still exist and
        // still carry the indexed identity. A dropped/repointed leaf is
        // skipped, so we never hand back a dead path.
        if let Some(leaf) = verify_entry(root, entry) {
            out.push(leaf);
        }
    }
    if out.is_empty() {
        // Either the index lacks this leaf (out-of-band add) or every
        // candidate was stale — only the full walk can distinguish a true
        // miss from an out-of-band hit, so defer to it.
        None
    } else {
        Some(out)
    }
}

/// Re-read the live `run.json` an index entry points at and confirm its
/// `run_id` still matches. Returns the verified [`Leaf`], or `None` if the
/// leaf is gone, unreadable, unparseable, or now holds a different identity.
fn verify_entry(root: &Path, entry: &IndexEntry) -> Option<Leaf> {
    let dir = root.join(&entry.rel_path);
    let bytes = fs::read(dir.join("run.json")).ok()?;
    let rec: RunRecord = serde_json::from_slice(&bytes).ok()?;
    if rec.run_id.to_hex() != entry.run_id {
        return None;
    }
    Some(Leaf { dir, record: rec })
}

/// Atomic + fsync write of `index.json`: write `index.json.tmp`, fsync it,
/// rename over `index.json`, fsync the directory — mirroring the store's
/// `run.json` durability ordering so concurrent writers never observe or
/// produce a torn index.
fn write_atomic(root: &Path, index: &CasIndex) -> std::io::Result<()> {
    if !root.exists() {
        fs::create_dir_all(root)?;
    }
    let json = serde_json::to_vec_pretty(index).map_err(std::io::Error::other)?;
    let tmp = root.join(INDEX_TMP);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&json)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, root.join(INDEX_FILE))?;
    // fsync the directory so the rename is durable.
    File::open(root)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runid::{ContentHash, RunStatus};
    use std::path::PathBuf;

    /// Minimal on-disk leaf: a directory under `root` holding a `run.json`
    /// with the given `run_id` and `kind`. Returns the leaf dir.
    fn plant_leaf(root: &Path, sub: &str, run_id: &str, kind: &str) -> PathBuf {
        let dir = root.join(sub);
        fs::create_dir_all(&dir).unwrap();
        let rec = format!(
            r#"{{
                "format_version": 1,
                "kind": "{kind}",
                "run_id": "{run_id}",
                "hash_version": 1,
                "ir_version": "0.7",
                "engine_version": "0.1.0+test",
                "levels": [
                    {{"name":"seed","label":"seed_1","hash":"{run_id}","schema_version":1}}
                ],
                "status": "completed",
                "artifacts": {{}},
                "provenance": {{"created_at":"2026-05-31T00:00:00Z","argv":[]}}
            }}"#
        );
        fs::write(dir.join("run.json"), rec).unwrap();
        dir
    }

    fn id(prefix: &str) -> String {
        format!("{:0<64}", prefix)
    }

    // Invariant 1: an out-of-band leaf added after the index was built must
    // still resolve. The index `resolve_prefix` returns None (it lacks the
    // leaf), forcing the caller to the walk — here we assert the index does
    // NOT claim a false "no match", and a rebuild then picks it up.
    #[test]
    fn out_of_band_leaf_is_a_miss_not_a_false_negative() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("aaaa1111"), "sim");
        let n = rebuild(root).unwrap();
        assert_eq!(n, 1, "one leaf indexed");

        // Add a second leaf out of band (index not updated).
        plant_leaf(root, "sims/b", &id("bbbb2222"), "sim");

        // The index cannot satisfy the new leaf's prefix: it must return a
        // miss (None), NOT an empty Some (which would be a false negative the
        // caller could not distinguish from a real hit set).
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, &id("bbbb2222")).is_none(),
            "out-of-band leaf must be a miss → caller falls back to walk"
        );
        // The known leaf still resolves from the index (exact run_id).
        let hit = resolve_prefix(root, ArtifactKind::Sim, &id("aaaa1111")).unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].run_id_hex(), id("aaaa1111"));

        // A rebuild (the repair) then finds both.
        let n = rebuild(root).unwrap();
        assert_eq!(n, 2, "rebuild discovers the out-of-band leaf");
        assert_eq!(resolve_prefix(root, ArtifactKind::Sim, &id("bbbb2222")).unwrap().len(), 1);
    }

    // Invariant 2: an index entry whose leaf was removed must NOT resolve to
    // the dead path. After rm -rf of the leaf, resolve_prefix on its run_id
    // must drop the stale entry (returning None — a miss to be re-walked),
    // never a Leaf at the nonexistent dir.
    #[test]
    fn stale_entry_is_dropped_not_resolved_to_dead_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dir = plant_leaf(root, "sims/a", &id("dead0001"), "sim");
        rebuild(root).unwrap();
        // The index can resolve it while it lives (exact run_id).
        assert!(resolve_prefix(root, ArtifactKind::Sim, &id("dead0001")).is_some());

        // Remove the leaf out of band; the index entry is now stale.
        fs::remove_dir_all(&dir).unwrap();

        // resolve_prefix must NOT return the dead path — the entry verifies
        // against the (now absent) run.json, fails, and is dropped → miss.
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, &id("dead0001")).is_none(),
            "a removed leaf must never resolve via the index to a dead path"
        );
    }

    // Invariant 2 variant: an entry repointed to a *different* identity (the
    // path now holds another run.json) must be dropped, not served.
    #[test]
    fn repointed_entry_is_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("1111aaaa"), "sim");
        rebuild(root).unwrap();
        // Overwrite the leaf's run.json with a different identity, leaving the
        // stale index entry pointing at run_id 1111aaaa at this path.
        plant_leaf(root, "sims/a", &id("9999ffff"), "sim");

        // The old indexed run_id no longer lives at that path → dropped.
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, &id("1111aaaa")).is_none(),
            "an entry whose path now holds a different identity must drop"
        );
    }

    // reindex rebuilds from run.json: add + remove out of band, rebuild, and
    // assert the index exactly matches the live tree (by run_id set).
    #[test]
    fn reindex_rebuilds_from_run_json() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("aa"), "sim");
        let removed = plant_leaf(root, "sims/b", &id("bb"), "sim");
        rebuild(root).unwrap();

        // Mutate the live tree out of band: remove b, add c.
        fs::remove_dir_all(&removed).unwrap();
        plant_leaf(root, "sims/c", &id("cc"), "sim");

        let n = rebuild(root).unwrap();
        assert_eq!(n, 2, "rebuild reflects the live tree (a + c, not b)");

        let idx = CasIndex::load(root).unwrap();
        let mut ids: Vec<&str> = idx.entries.iter().map(|e| e.run_id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, vec![id("aa"), id("cc")]);
        // b is gone — its run_id must not appear.
        assert!(!idx.entries.iter().any(|e| e.run_id == id("bb")));
    }

    // A malformed index.json is a clean miss (None on load), never a panic.
    #[test]
    fn malformed_index_is_a_clean_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("abcd"), "sim");
        fs::write(root.join(INDEX_FILE), b"{ this is not json").unwrap();

        assert!(CasIndex::load(root).is_none(), "malformed index loads as None");
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, &id("abcd")).is_none(),
            "malformed index → miss → caller walks (no panic)"
        );
    }

    // An absent index.json is a clean miss, never a panic.
    #[test]
    fn absent_index_is_a_clean_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        assert!(CasIndex::load(root).is_none());
        assert!(resolve_prefix(root, ArtifactKind::Sim, &id("abcd")).is_none());
    }

    // Ambiguity safety (the exact-id-only fast path): even when the index HAS
    // a matching entry, a short prefix must NOT be served from the index —
    // because an un-indexed out-of-band leaf could share that prefix, and
    // trusting the index would silently under-report ambiguity. Only an exact
    // 64-hex run_id (unique → ambiguity impossible) uses the fast path; short
    // prefixes fall through (None) to the authoritative walk.
    #[test]
    fn short_prefix_is_never_trusted_by_the_index() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("abcd1111"), "sim");
        rebuild(root).unwrap();

        // Exact full run_id → fast-path hit.
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, &id("abcd1111")).is_some(),
            "an exact 64-hex run_id must use the index fast path"
        );
        // Short prefix that DOES match the indexed entry → still declined, so
        // the caller walks and can detect a colliding un-indexed leaf.
        assert!(
            resolve_prefix(root, ArtifactKind::Sim, "abcd1111").is_none(),
            "a short prefix must never be trusted by the index (ambiguity safety)"
        );
    }

    // A version mismatch is a clean miss (forward/back schema break).
    #[test]
    fn version_mismatch_is_a_clean_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let bogus = serde_json::json!({ "version": 999, "entries": [] });
        fs::write(root.join(INDEX_FILE), serde_json::to_vec(&bogus).unwrap()).unwrap();
        assert!(CasIndex::load(root).is_none(), "future version → clean miss");
    }

    // Atomic write leaves no index.json.tmp behind after success.
    #[test]
    fn write_is_atomic_no_tmp_left() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("0a0a"), "sim");
        rebuild(root).unwrap();
        assert!(root.join(INDEX_FILE).exists(), "index.json written");
        assert!(
            !root.join(INDEX_TMP).exists(),
            "no index.json.tmp left after a successful atomic write"
        );
    }

    // The index never claims a leaf the live walk would not: a kind filter is
    // honoured (a Sim prefix does not match a FitStage leaf of the same id).
    #[test]
    fn kind_filter_is_honoured() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // Two leaves sharing the SAME run_id hex but different kinds, so the
        // exact-id query exercises the kind filter (not the id alone).
        plant_leaf(root, "sims/a", &id("c0ffee"), "sim");
        plant_leaf(root, "fits/b", &id("c0ffee"), "fit_stage");
        rebuild(root).unwrap();
        // Querying the Sim kind must not return the fit_stage leaf even though
        // its run_id is identical.
        let sims = resolve_prefix(root, ArtifactKind::Sim, &id("c0ffee")).unwrap();
        assert_eq!(sims.len(), 1);
        assert_eq!(sims[0].record.kind, ArtifactKind::Sim);
    }

    // Smoke: ContentHash round-trips through the index entry's hex string, so
    // a resolved leaf's run_id matches what was indexed.
    #[test]
    fn indexed_run_id_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        plant_leaf(root, "sims/a", &id("feedface"), "sim");
        rebuild(root).unwrap();
        let leaf = &resolve_prefix(root, ArtifactKind::Sim, &id("feedface")).unwrap()[0];
        let parsed = ContentHash::from_hex(&leaf.run_id_hex()).unwrap();
        assert_eq!(parsed.to_hex(), id("feedface"));
        // status round-trips too.
        let idx = CasIndex::load(root).unwrap();
        assert_eq!(idx.entries[0].status, RunStatus::Completed);
    }
}
