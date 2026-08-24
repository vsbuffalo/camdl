//! `CasStore` integration tests: lookup outcomes (incl. PathPrefixCollision),
//! the exact-set manifest, Mode A atomic commit + collision disambiguation,
//! and Mode B `Running → Completed` with the `O_EXCL` claim.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::hash::{ContentHash, HASH_VERSION};
use crate::kind::ArtifactKind;
use crate::record::{Provenance, RunStatus, FORMAT_VERSION};

fn nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
}

/// A fresh temp root, removed by `cleanup`.
fn tmp_root(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("runid_store_{}_{}_{}", tag, std::process::id(), nanos()));
    fs::create_dir_all(&p).unwrap();
    p
}

fn cleanup(p: &PathBuf) {
    fs::remove_dir_all(p).ok();
}

fn id(byte: u8) -> ContentHash {
    ContentHash::from_bytes([byte; 32])
}

fn record(run_id: ContentHash) -> RunRecord {
    RunRecord {
        format_version: FORMAT_VERSION,
        kind: ArtifactKind::Sim,
        run_id,
        hash_version: HASH_VERSION,
        ir_version: "0.7".into(),
        engine_version: "0.3.0".into(),
        levels: vec![],
        deps: vec![],
        status: RunStatus::Running,
        artifacts: BTreeMap::new(),
        output_schema: BTreeMap::new(),
        children: BTreeMap::new(),
        inputs: serde_json::Value::Null,
        provenance: Provenance::default(),
    }
}

fn arts(content: &[u8]) -> Artifacts {
    let mut a = Artifacts::new();
    a.insert("traj.tsv", content.to_vec());
    a
}

// ── lookup outcomes ──────────────────────────────────────────────────────────

#[test]
fn lookup_miss_on_absent() {
    let root = tmp_root("miss");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    assert!(matches!(store.lookup(&leaf, &LeafIdentity::new(id(1))), Lookup::Miss));
    cleanup(&root);
}

#[test]
fn commit_then_lookup_hit() {
    let root = tmp_root("hit");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    let dest = store.commit_atomic(&leaf, record(a), arts(b"S\tI\n999\t1\n")).unwrap();
    assert_eq!(dest, leaf, "no collision → lands at the intended path");
    match store.lookup(&dest, &LeafIdentity::new(a)) {
        Lookup::Hit(r) => {
            assert_eq!(r.run_id, a);
            assert_eq!(r.status, RunStatus::Completed);
            assert!(r.artifacts.contains_key("traj.tsv"));
        }
        other => panic!("expected Hit, got {other:?}"),
    }
    assert!(dest.join("traj.tsv").exists());
    cleanup(&root);
}

// ── S4: the augment door ────────────────────────────────────────────────────

/// A completed leaf can gain an artifact. Before this the store had no such
/// operation, so `simulate --event-log` against an existing leaf staged the
/// log and had the whole set discarded — the log silently lost.
#[test]
fn augment_adds_an_artifact_to_a_completed_leaf() {
    let root = tmp_root("augment");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    let dest = store.commit_atomic(&leaf, record(a), arts(b"S\tI\n9\t1\n")).unwrap();

    store.augment(&dest, &LeafIdentity::new(a), "event_log.tsv", b"t\tevent\n").unwrap();

    // The file is there, the record names it, and the leaf is still a Hit —
    // i.e. the added file is part of the exact set, not an orphan.
    assert_eq!(fs::read(dest.join("event_log.tsv")).unwrap(), b"t\tevent\n");
    match store.lookup(&dest, &LeafIdentity::new(a)) {
        Lookup::Hit(r) => assert!(r.artifacts.contains_key("event_log.tsv"),
            "the augmented file must be in the manifest, or the exact-set scan \
             reports it as an orphan and the leaf goes Stale"),
        other => panic!("expected Hit after augment, got {other:?}"),
    }
    cleanup(&root);
}

/// Re-adding the same bytes is a no-op, so a rerun that records the same log
/// again is harmless.
#[test]
fn augment_is_idempotent_for_identical_bytes() {
    let root = tmp_root("augmentidem");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    let dest = store.commit_atomic(&leaf, record(a), arts(b"traj")).unwrap();
    let idl = LeafIdentity::new(a);
    store.augment(&dest, &idl, "event_log.tsv", b"same").unwrap();
    store.augment(&dest, &idl, "event_log.tsv", b"same").unwrap();
    assert!(matches!(store.lookup(&dest, &idl), Lookup::Hit(_)));
    cleanup(&root);
}

/// Re-adding DIFFERENT bytes under one name is the same identity bug the
/// commit path refuses: same key must mean same bytes.
#[test]
fn augment_refuses_divergent_bytes() {
    let root = tmp_root("augmentdiverge");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    let dest = store.commit_atomic(&leaf, record(a), arts(b"traj")).unwrap();
    let idl = LeafIdentity::new(a);
    store.augment(&dest, &idl, "event_log.tsv", b"first").unwrap();
    let err = store.augment(&dest, &idl, "event_log.tsv", b"second").unwrap_err();
    match &err {
        CasError::DivergentRecompute { file, .. } => assert_eq!(file, "event_log.tsv"),
        other => panic!("expected DivergentRecompute, got {other:?}"),
    }
    assert_eq!(fs::read(dest.join("event_log.tsv")).unwrap(), b"first",
        "the incumbent artifact must be untouched");
    cleanup(&root);
}

/// Never augment somebody else's artifact, and never one a live process holds.
#[test]
fn augment_respects_identity_and_live_holders() {
    let root = tmp_root("augmentguard");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    store.commit_atomic(&leaf, record(id(0xaa)), arts(b"traj")).unwrap();

    // Different identity at this path → refused, nothing written.
    let err = store.augment(&leaf, &LeafIdentity::new(id(0xbb)), "x.tsv", b"x").unwrap_err();
    assert!(matches!(err, CasError::AlreadyCompleted { .. }), "got {err:?}");
    assert!(!leaf.join("x.tsv").exists());

    // Live holder → refused.
    fs::write(leaf.join(".lock"), std::process::id().to_string()).unwrap();
    let err = store.augment(&leaf, &LeafIdentity::new(id(0xaa)), "y.tsv", b"y").unwrap_err();
    assert!(matches!(err, CasError::FitInProgress { .. }), "got {err:?}");
    assert!(!leaf.join("y.tsv").exists());
    cleanup(&root);
}

/// Augmenting something that is not a completed leaf of this identity is a
/// no-op, not an error: the caller's next run writes it from scratch.
#[test]
fn augment_on_a_missing_leaf_is_a_no_op() {
    let root = tmp_root("augmentmiss");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    store.augment(&leaf, &LeafIdentity::new(id(0xaa)), "event_log.tsv", b"x").unwrap();
    assert!(!leaf.join("event_log.tsv").exists());
    cleanup(&root);
}

// ── S2: the overwrite door ──────────────────────────────────────────────────

/// `--force` had no store-level meaning: there was no path that replaced a
/// Completed leaf, so batch recomputed and had its bytes discarded at commit,
/// while fit and survey died with AlreadyCompleted. `displace_completed`
/// quarantines the incumbent so the recompute lands on a clean leaf — and
/// quarantine, not delete, so a forced rerun never destroys what it replaces.
#[test]
fn displace_completed_quarantines_the_incumbent_and_frees_the_path() {
    let root = tmp_root("displace");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    store.commit_atomic(&leaf, record(a), arts(b"FIRST")).unwrap();

    store.displace_completed(&leaf, &LeafIdentity::new(a)).unwrap();
    assert!(matches!(store.lookup(&leaf, &LeafIdentity::new(a)), Lookup::Miss),
        "the displaced leaf must no longer be a cache hit");
    // Preserved as evidence, not deleted.
    let q = root.join(".quarantine");
    assert!(fs::read_dir(&q).map(|d| d.count()).unwrap_or(0) >= 1,
        "the incumbent must be quarantined, not destroyed");

    // The recompute now lands cleanly — and with different bytes, which is the
    // whole point of forcing (pre-S1 this was a silent discard, post-S1 it
    // would be DivergentRecompute without the displace).
    let dest = store.commit_atomic(&leaf, record(a), arts(b"SECOND")).unwrap();
    assert_eq!(fs::read(dest.join("traj.tsv")).unwrap(), b"SECOND");
    cleanup(&root);
}

/// Forcing must never displace somebody else's artifact: a leaf holding a
/// DIFFERENT identity at this path is left untouched.
#[test]
fn displace_completed_leaves_a_different_identity_alone() {
    let root = tmp_root("displaceother");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    store.commit_atomic(&leaf, record(id(0xaa)), arts(b"THEIRS")).unwrap();
    // Force under a different identity — a no-op.
    store.displace_completed(&leaf, &LeafIdentity::new(id(0xbb))).unwrap();
    assert_eq!(fs::read(leaf.join("traj.tsv")).unwrap(), b"THEIRS",
        "a different identity's leaf must survive a force");
    cleanup(&root);
}

/// And never displace a leaf a live process is holding.
#[test]
fn displace_completed_refuses_a_live_holder() {
    let root = tmp_root("displacelive");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    store.commit_atomic(&leaf, record(a), arts(b"FIRST")).unwrap();
    fs::write(leaf.join(".lock"), std::process::id().to_string()).unwrap();
    let err = store.displace_completed(&leaf, &LeafIdentity::new(a)).unwrap_err();
    assert!(matches!(err, CasError::FitInProgress { .. }), "got {err:?}");
    cleanup(&root);
}

// ── S3: the claim guard ─────────────────────────────────────────────────────

/// A claim dropped without `finalize` marks its leaf `Failed` and releases the
/// lock. Before the guard the leaf stayed `Running` with a live `.lock`, so a
/// cleanly-failed run looked identical to a `kill -9` and the next claimant
/// had to wait for PID-liveness reclaim.
#[test]
fn abandoned_claim_marks_the_leaf_failed_and_drops_the_lock() {
    let root = tmp_root("abandoned");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    {
        let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
        claim.write("partial.tsv", b"half").unwrap();
        // dropped here without finalize — the error/`?`/unwind path
    }
    let rec = match read_record(&leaf) {
        ReadResult::Ok(r) => r,
        ReadResult::Absent => panic!("run.json missing after an abandoned claim"),
        ReadResult::Unparseable => panic!("run.json unparseable after an abandoned claim"),
    };
    assert_eq!(rec.status, RunStatus::Failed,
        "an abandoned claim must record Failed, not leave Running");
    assert!(!leaf.join(".lock").exists(), "the lock must be released");
    // And the leaf is reclaimable: a later run of the same identity recomputes.
    assert!(matches!(
        store.lookup(&leaf, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Incomplete)
    ));
    cleanup(&root);
}

/// The guard must not touch a finalized leaf.
#[test]
fn finalized_claim_is_untouched_by_the_guard() {
    let root = tmp_root("finalized");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("traj.tsv", b"S\tI\n9\t1\n").unwrap();
    let dest = claim.finalize(record(id(0xaa))).unwrap();
    match store.lookup(&dest, &LeafIdentity::new(id(0xaa))) {
        Lookup::Hit(r) => assert_eq!(r.status, RunStatus::Completed),
        other => panic!("expected Hit after finalize, got {other:?}"),
    }
    cleanup(&root);
}

// ── S1: divergence check at commit ──────────────────────────────────────────
// Proposal: docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md

/// Same identity, same bytes → benign dedup, exactly as before S1.
#[test]
fn identical_recompute_still_dedups() {
    let root = tmp_root("dedup");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    let d1 = store.commit_atomic(&leaf, record(a), arts(b"same bytes")).unwrap();
    let d2 = store.commit_atomic(&leaf, record(a), arts(b"same bytes")).unwrap();
    assert_eq!(d1, d2, "identical recompute returns the incumbent leaf");
    cleanup(&root);
}

/// Same identity, DIFFERENT bytes → the staged recompute disagrees with the
/// incumbent. Runs are seeded-deterministic, so this is an identity bug (a
/// knob missing from the run_id) or nondeterminism — either must be loud,
/// never a silent discard. This is the check that would have caught the
/// `--integrator` class at first occurrence.
#[test]
fn divergent_recompute_is_a_loud_error_not_a_silent_discard() {
    let root = tmp_root("diverge");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    store.commit_atomic(&leaf, record(a), arts(b"first result")).unwrap();
    let err = store
        .commit_atomic(&leaf, record(a), arts(b"second, different result"))
        .unwrap_err();
    match &err {
        CasError::DivergentRecompute { file, .. } => assert_eq!(file, "traj.tsv"),
        other => panic!("expected DivergentRecompute, got {other:?}"),
    }
    // The incumbent is untouched — divergence never clobbers.
    assert_eq!(fs::read(leaf.join("traj.tsv")).unwrap(), b"first result");
    // The staged bytes are quarantined as evidence, not deleted.
    let q = root.join(".quarantine");
    let quarantined = fs::read_dir(&q)
        .map(|d| d.count())
        .unwrap_or(0);
    assert!(quarantined >= 1, "staged dir must be preserved under .quarantine");
    cleanup(&root);
}

/// Same identity, staged strict superset with equal shared bytes — the
/// "leaf gains an artifact" case (e.g. --event-log against a pre-existing
/// leaf). Until the S4 augment door lands this dedups (the extra file is
/// dropped), but it must NOT be a divergence error.
#[test]
fn superset_recompute_with_equal_shared_bytes_dedups() {
    let root = tmp_root("superset");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let a = id(0xaa);
    store.commit_atomic(&leaf, record(a), arts(b"traj bytes")).unwrap();
    let mut plus = arts(b"traj bytes");
    plus.insert("event_log.tsv", b"t\tevent\n".to_vec());
    let dest = store.commit_atomic(&leaf, record(a), plus).unwrap();
    assert_eq!(dest, leaf);
    assert!(!leaf.join("event_log.tsv").exists(),
        "pre-S4: the extra artifact is not adopted (and not an error)");
    cleanup(&root);
}

#[test]
fn lookup_collision_on_different_identity() {
    let root = tmp_root("coll");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    store.commit_atomic(&leaf, record(id(0xaa)), arts(b"a")).unwrap();
    // Look up with a DIFFERENT expected identity at the same path.
    match store.lookup(&leaf, &LeafIdentity::new(id(0xbb))) {
        Lookup::Collision(r) => assert_eq!(r.run_id, id(0xaa), "incumbent surfaced, untouched"),
        other => panic!("expected Collision, got {other:?}"),
    }
    cleanup(&root);
}

#[test]
fn lookup_stale_incomplete_on_running() {
    let root = tmp_root("running");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    // A claimed-but-not-finalized leaf is Running.
    let _claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    assert!(matches!(
        store.lookup(&leaf, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Incomplete)
    ));
    cleanup(&root);
}

#[test]
fn lookup_stale_corrupt_on_missing_file() {
    let root = tmp_root("missing");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let dest = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"data")).unwrap();
    fs::remove_file(dest.join("traj.tsv")).unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Corrupt)
    ));
    cleanup(&root);
}

#[test]
fn lookup_stale_orphan_on_unlisted_file() {
    let root = tmp_root("orphan");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let dest = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"data")).unwrap();
    fs::write(dest.join("stray.txt"), b"crash debris").unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::OrphanFiles)
    ));
    cleanup(&root);
}

#[test]
fn lookup_stale_corrupt_on_truncated_run_json() {
    let root = tmp_root("trunc");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let dest = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"data")).unwrap();
    fs::write(dest.join("run.json"), b"{ truncated").unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Corrupt)
    ));
    cleanup(&root);
}

#[test]
fn declared_child_subdir_is_not_an_orphan() {
    let root = tmp_root("child");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let mut rec = record(id(0xaa));
    rec.children.insert("obs".into(), vec![id(0xcc)]);
    let dest = store.commit_atomic(&leaf, rec, arts(b"data")).unwrap();
    // A declared child namespace subdir is recognized, not orphaned.
    fs::create_dir_all(dest.join("obs")).unwrap();
    assert!(matches!(store.lookup(&dest, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    // An UNdeclared subdir, however, is an orphan.
    fs::create_dir_all(dest.join("junk")).unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::OrphanFiles)
    ));
    cleanup(&root);
}

// ── Mode A: collisions and lost races ────────────────────────────────────────

#[test]
fn pathprefix_collision_disambiguates_without_data_loss() {
    let root = tmp_root("disambig");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");

    // Two distinct full identities forced onto the SAME 8-char segment path.
    let pa = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"AAAA")).unwrap();
    let pb = store.commit_atomic(&leaf, record(id(0xbb)), arts(b"BBBB")).unwrap();

    assert_eq!(pa, leaf, "first identity takes the base path");
    assert_ne!(pb, leaf, "second identity is disambiguated");
    assert!(pb.file_name().unwrap().to_string_lossy().contains('~'), "disambiguator appended");

    // The incumbent's bytes are untouched; both lookups resolve.
    assert_eq!(fs::read(pa.join("traj.tsv")).unwrap(), b"AAAA");
    assert_eq!(fs::read(pb.join("traj.tsv")).unwrap(), b"BBBB");
    assert!(matches!(store.lookup(&pa, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    assert!(matches!(store.lookup(&pb, &LeafIdentity::new(id(0xbb))), Lookup::Hit(_)));
    cleanup(&root);
}

#[test]
fn lost_race_returns_incumbent_without_overwriting() {
    let root = tmp_root("race");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let p1 = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"ORIGINAL")).unwrap();
    // A second identical-identity, identical-bytes commit finds the completed
    // leaf → benign race, same destination, incumbent untouched. (Pre-S1 this
    // test used DIFFERENT bytes to observe the non-overwrite; different bytes
    // under one run_id are now the DivergentRecompute error — see
    // divergent_recompute_is_a_loud_error_not_a_silent_discard.)
    let p2 = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"ORIGINAL")).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(fs::read(p1.join("traj.tsv")).unwrap(), b"ORIGINAL");
    cleanup(&root);
}

#[test]
fn commit_clears_same_identity_stale_orphans() {
    let root = tmp_root("clearstale");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    // Leave a same-identity stale (Running) leaf with an orphan file.
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("partial_chain.tsv", b"half").unwrap();
    fs::remove_file(leaf.join(".lock")).unwrap(); // simulate a crashed run

    // Mode A commit of the same identity clears the stale leaf and recomputes.
    let dest = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"clean")).unwrap();
    assert_eq!(dest, leaf);
    assert!(!dest.join("partial_chain.tsv").exists(), "crashed orphan must be cleared");
    assert_eq!(fs::read(dest.join("traj.tsv")).unwrap(), b"clean");
    cleanup(&root);
}

#[test]
fn concurrent_same_identity_commits_are_race_safe() {
    // The dedup'd-draw-row case: many threads commit the SAME identity (same
    // content) to the same leaf concurrently. Each must succeed (no spurious
    // error from a clobbered staging dir or a lost rename race), and the leaf
    // must be a single intact Completed artifact.
    let root = tmp_root("concurrent");
    let store = FsCasStore::new(&root);
    let leaf = root.join("sims").join("sir-aaaaaaaa");
    let ident = id(0xaa);
    let content: Vec<u8> = b"deterministic-output-for-one-identity".to_vec();

    // A barrier releases all threads into commit_atomic at the same instant,
    // maximizing staging + rename contention — the old shared-staging
    // (`.staging/{run_id}` + remove_dir_all) code clobbers under this.
    const N: usize = 8;
    let barrier = std::sync::Barrier::new(N);
    let results: Vec<Result<PathBuf, CasError>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..N)
            .map(|_| {
                let store = &store;
                let leaf = &leaf;
                let barrier = &barrier;
                let content = content.clone();
                s.spawn(move || {
                    barrier.wait();
                    store.commit_atomic(leaf, record(ident), arts(&content))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Every commit succeeded and resolved to the one leaf (same identity → no
    // disambiguation).
    for r in &results {
        let dest = r.as_ref().unwrap_or_else(|e| panic!("commit errored under concurrency: {e}"));
        assert_eq!(dest, &leaf);
    }
    // The committed leaf is an intact Hit with the expected bytes.
    match store.lookup(&leaf, &LeafIdentity::new(ident)) {
        Lookup::Hit(_) => assert_eq!(fs::read(leaf.join("traj.tsv")).unwrap(), content),
        other => panic!("expected Hit, got {other:?}"),
    }
    cleanup(&root);
}

// ── Mode B: streamed Running → Completed ─────────────────────────────────────

#[test]
fn mode_b_claim_stream_finalize_then_hit() {
    let root = tmp_root("modeb");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    // Visible + Running during the stream.
    assert!(matches!(
        store.lookup(&leaf, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Incomplete)
    ));
    claim.write("chain_0.tsv", b"theta\n1.0\n").unwrap();
    let dest = claim.finalize(record(id(0xaa))).unwrap();

    match store.lookup(&dest, &LeafIdentity::new(id(0xaa))) {
        Lookup::Hit(r) => assert!(r.artifacts.contains_key("chain_0.tsv")),
        other => panic!("expected Hit, got {other:?}"),
    }
    assert!(!dest.join(".lock").exists(), "lock removed at finalize");
    cleanup(&root);
}

#[test]
fn mode_b_second_claim_fails_fast() {
    let root = tmp_root("excl");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let _held = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    // A concurrent identical fit must fail fast on the O_EXCL claim, not
    // interleave bytes into the shared chain files.
    match store.claim_streaming(&leaf, record(id(0xaa))) {
        Err(CasError::FitInProgress { pid, .. }) => assert_eq!(pid, std::process::id()),
        other => panic!("expected FitInProgress, got {other:?}"),
    }
    cleanup(&root);
}

#[test]
fn mode_b_reclaims_stale_lock_held_by_dead_pid() {
    // Serialize against the gap-hook test: an installed hook is a process-global.
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = tmp_root("deadpid");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("orphan_chain.tsv", b"partial").unwrap();
    // A fit killed mid-run (Ctrl-C / SIGPIPE / OOM / crash) never finalizes, so
    // its `.lock` + Running run.json linger — but the holder PID is dead. Plant
    // a definitely-dead PID (a child we spawn and immediately reap).
    let mut child = std::process::Command::new("true").spawn().expect("spawn true");
    let dead_pid = child.id();
    child.wait().expect("reap");
    fs::write(leaf.join(".lock"), dead_pid.to_string()).unwrap();

    // The dead-PID lock is stale → reclaim (clear orphans + re-claim), NOT
    // FitInProgress. A live holder still blocks (mode_b_second_claim_fails_fast).
    let claim2 = store.claim_streaming(&leaf, record(id(0xaa)))
        .expect("a lock held by a dead PID must be reclaimed, not FitInProgress");
    assert!(!leaf.join("orphan_chain.tsv").exists(), "stale orphan cleared on dead-PID reclaim");
    let dest = claim2.finalize(record(id(0xaa))).unwrap();
    assert!(matches!(store.lookup(&dest, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    cleanup(&root);
}

/// A process that dies holding the `.reclaim` serializer used to wedge the
/// leaf permanently: every later reclaim saw `AlreadyExists` and refused, so
/// the leaf could never be recomputed without a manual `rm`. A stranded
/// serializer (dead owner pid) is now cleared at the point of contention.
#[test]
fn stranded_reclaim_serializer_does_not_wedge_the_leaf() {
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = tmp_root("strandedreclaim");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("orphan_chain.tsv", b"partial").unwrap();

    // Two dead pids: one holding `.lock` (the crashed run), one holding
    // `.reclaim` (a second process that died mid-takeover).
    let mut c1 = std::process::Command::new("true").spawn().expect("spawn");
    let dead_lock_pid = c1.id();
    c1.wait().expect("reap");
    let mut c2 = std::process::Command::new("true").spawn().expect("spawn");
    let dead_reclaim_pid = c2.id();
    c2.wait().expect("reap");
    fs::write(leaf.join(".lock"), dead_lock_pid.to_string()).unwrap();
    fs::write(leaf.join(".reclaim"), dead_reclaim_pid.to_string()).unwrap();

    let claim2 = store.claim_streaming(&leaf, record(id(0xaa)))
        .expect("a .reclaim stranded by a dead process must not wedge the leaf");
    let dest = claim2.finalize(record(id(0xaa))).unwrap();
    assert!(matches!(store.lookup(&dest, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    cleanup(&root);
}

/// The converse: a `.reclaim` held by a LIVE process is a genuine concurrent
/// reclaim and must still refuse — the stranded-clearing must not become a
/// way to barge into the critical section.
#[test]
fn live_reclaim_serializer_still_refuses() {
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = tmp_root("livereclaim");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("orphan_chain.tsv", b"partial").unwrap();
    let mut c1 = std::process::Command::new("true").spawn().expect("spawn");
    let dead_lock_pid = c1.id();
    c1.wait().expect("reap");
    fs::write(leaf.join(".lock"), dead_lock_pid.to_string()).unwrap();
    // OUR pid is alive by construction.
    fs::write(leaf.join(".reclaim"), std::process::id().to_string()).unwrap();

    let err = store.claim_streaming(&leaf, record(id(0xaa)))
        .expect_err("a live .reclaim holder must still block the takeover");
    assert!(matches!(err, CasError::FitInProgress { .. }), "got {err:?}");
    cleanup(&root);
}

#[test]
fn mode_b_reclaims_stale_running_when_unlocked() {
    let root = tmp_root("reclaim");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("orphan_chain.tsv", b"partial").unwrap();
    // Simulate a crash: the lock is gone but a stale Running run.json remains.
    fs::remove_file(leaf.join(".lock")).unwrap();

    let claim2 = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    assert!(!leaf.join("orphan_chain.tsv").exists(), "crashed orphan cleared on reclaim");
    let dest = claim2.finalize(record(id(0xaa))).unwrap();
    assert!(matches!(store.lookup(&dest, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    cleanup(&root);
}

/// Deterministic proof of the remove→recreate TOCTOU in `reclaim_or_refuse`.
///
/// The `RECLAIM_GAP_HOOK` fires at exactly the point the buggy code left `.lock`
/// absent (between removing the dead `.lock` and recreating it), and drives a
/// real bare claimant through a full claim→write→finalize cycle there. On the
/// buggy code the intruder's `create_new(.lock)` succeeds into the open gap and
/// `finalize` removes `.lock`; the reclaimer then resumes, its
/// `create_new(.lock)` succeeds against the absent lock, and it re-enters the
/// critical section and `clear_except_lock`-wipes the intruder's just-completed
/// result → **two** finalizes of the same leaf. Under the atomic-rename fix the
/// intruder's `create_new(.lock)` always fails (`.lock` never absent) → it
/// routes through reclaim → `.reclaim` held → refuses, and only the reclaimer
/// wins.
///
/// The true contract — independent of which claimant wins — is **exactly one**
/// successful finalize and an intact `Completed` leaf. Buggy: two successes
/// (FAIL, every run). Fixed: one (PASS).
#[test]
fn mode_b_reclaim_gap_is_not_exploitable() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    let root = tmp_root("reclaim_gap");
    let store = Arc::new(FsCasStore::new(&root));
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");

    // A definitely-dead PID for the planted crashed lock.
    let mut child = std::process::Command::new("true").spawn().expect("spawn true");
    let dead_pid = child.id();
    child.wait().expect("reap");

    // Plant a crashed run: a `Running` leaf + orphan whose `.lock` holder is dead.
    fs::remove_dir_all(&leaf).ok();
    let c = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    c.write("orphan_chain.tsv", b"partial").unwrap();
    drop(c);
    fs::write(leaf.join(".lock"), dead_pid.to_string()).unwrap();

    // Counts every claimant that finalizes a leaf. The contract is exactly one.
    let finalizes = Arc::new(AtomicUsize::new(0));
    let fired = Arc::new(AtomicBool::new(false));
    // True iff the intruder's claim was refused — the atomic-rename fix's direct
    // signature (`.lock` never absent ⇒ the bare `create_new(.lock)` fails ⇒
    // reclaim ⇒ `.reclaim` held ⇒ refuse). Isolates the rename fix from the
    // defense-in-depth guard, which would otherwise mask a still-open window.
    let intruder_refused = Arc::new(AtomicBool::new(false));

    // The gap hook is a process-global; serialize against the other tests that
    // walk `reclaim_or_refuse` so an installed hook never fires in their
    // context. Held for the whole test.
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // RAII disarm: restore the hook to `None` on every exit path (incl. an
    // assertion unwind), so it can never leak into another test.
    struct DisarmOnDrop;
    impl Drop for DisarmOnDrop {
        fn drop(&mut self) {
            *super::RECLAIM_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    let _disarm = DisarmOnDrop;

    // Install the gap hook: exactly once, drive a real bare claimant through a
    // full claim→write→finalize on a joined thread, so its whole cycle
    // completes before the reclaimer resumes. The claim is tolerant of refusal
    // (the correct behaviour under the fix): only a successful claim finalizes.
    {
        let store_h = Arc::clone(&store);
        let leaf_h = leaf.clone();
        let fin_h = Arc::clone(&finalizes);
        let fired_h = Arc::clone(&fired);
        let refused_h = Arc::clone(&intruder_refused);
        let mut guard = super::RECLAIM_GAP_HOOK.lock().unwrap();
        *guard = Some(Box::new(move |_lock: &std::path::Path| {
            if fired_h.swap(true, Ordering::SeqCst) {
                return; // fire only on the first reclaim
            }
            let leaf2 = leaf_h.clone();
            let store2 = Arc::clone(&store_h);
            let fin2 = Arc::clone(&fin_h);
            let refused2 = Arc::clone(&refused_h);
            std::thread::spawn(move || {
                match store2.claim_streaming(&leaf2, record(id(0xaa))) {
                    Ok(claim) => {
                        claim
                            .write("chain_intruder/trace.tsv", b"sweep\tll\n1\t-1.0\n")
                            .unwrap();
                        claim.finalize(record(id(0xaa))).unwrap();
                        fin2.fetch_add(1, Ordering::SeqCst);
                    }
                    // Correct: the window is closed, the intruder is refused.
                    Err(CasError::FitInProgress { .. }) | Err(CasError::AlreadyCompleted { .. }) => {
                        refused2.store(true, Ordering::SeqCst);
                    }
                    Err(e) => panic!("intruder hit an unexpected error: {e:?}"),
                }
            })
            .join()
            .unwrap();
        }));
    }

    // The reclaimer runs. If it wins the leaf it finalizes (as production code
    // would); a refusal is also acceptable. The bug is *both* finalizing.
    let reclaimer = store.claim_streaming(&leaf, record(id(0xaa)));

    // Disarm now (the RAII guard is the panic-safe backstop).
    *super::RECLAIM_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;

    match reclaimer {
        Ok(claim) => {
            claim
                .write("chain_reclaimer/trace.tsv", b"sweep\tll\n1\t-2.0\n")
                .unwrap();
            claim.finalize(record(id(0xaa))).unwrap();
            finalizes.fetch_add(1, Ordering::SeqCst);
        }
        Err(CasError::FitInProgress { .. }) | Err(CasError::AlreadyCompleted { .. }) => {}
        // The defense-in-depth guard turns a residual race into this loud error
        // rather than a silent clobber — also acceptable (no double-write).
        Err(CasError::ReclaimRaceCompleted { .. }) => {}
        Err(e) => panic!("unexpected reclaim error: {e:?}"),
    }

    assert!(fired.load(Ordering::SeqCst), "the gap hook never fired — the reclaim path was not exercised");
    // The rename fix's direct signature: a bare claimant arriving in what was
    // the gap is refused, because `.lock` is never observably absent.
    assert!(
        intruder_refused.load(Ordering::SeqCst),
        "the intruder slipped into the reclaim gap (`.lock` was observably absent) — the window is still open"
    );
    assert_eq!(
        finalizes.load(Ordering::SeqCst),
        1,
        "exactly one claimant must finalize the leaf — two means the reclaim gap was exploited (double-write)"
    );
    // The surviving leaf must be a single intact Completed Hit.
    assert!(
        matches!(store.lookup(&leaf, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)),
        "the winning claimant's Completed leaf must be intact, not clobbered"
    );
    cleanup(&root);
}

/// Deterministic proof of the `Lookup::Miss` false-orphan race in
/// `resolve_claim_dir`.
///
/// When a claimant wins the `.lock` and runs `clear_except_lock`, the leaf
/// holds its live `.lock` but `run.json` is transiently absent (just cleared,
/// not yet rewritten). A *concurrent* claimant that reads the leaf in that
/// window sees `Lookup::Miss` ("dir exists, no run.json") and — on the buggy
/// code — treats it as orphan debris: it `rename`s the whole dir into
/// quarantine, out from under the active holder. The holder's next write
/// (`write_record_atomic`'s `rename(run.json.tmp → run.json)`) then fails with
/// `Io(NotFound)` because its directory was moved — the exact symptom seen
/// under CI concurrency (`unexpected claim error: Io(NotFound)`).
///
/// The `CLEAR_GAP_HOOK` fires at exactly that window and drives a real
/// concurrent claimant of the SAME identity through `claim_streaming` there.
/// Under the fix the intruder sees the live `.lock` and routes to the lock
/// gate → `FitInProgress` (refused, no quarantine). The holder then completes
/// cleanly. Buggy: the holder's finalize errors / the leaf is clobbered.
#[test]
fn mode_b_clear_window_is_not_quarantined() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // The clear-gap hook is a process-global; serialize against the other
    // hook-installing tests (they share RECLAIM_HOOK_TEST_LOCK).
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let root = tmp_root("clear_window");
    let store = Arc::new(FsCasStore::new(&root));
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");

    // A definitely-dead PID for the planted crashed lock, so the winner
    // reclaims it (→ enters claim_streaming → clear_except_lock → hook).
    let mut child = std::process::Command::new("true").spawn().expect("spawn true");
    let dead_pid = child.id();
    child.wait().expect("reap");

    // Plant a crashed run: a `Running` leaf + orphan whose `.lock` holder is dead.
    fs::remove_dir_all(&leaf).ok();
    let c = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    c.write("orphan_chain.tsv", b"partial").unwrap();
    drop(c);
    fs::write(leaf.join(".lock"), dead_pid.to_string()).unwrap();

    let fired = Arc::new(AtomicBool::new(false));
    // True iff the intruder was refused (correct: a live `.lock` holds the
    // leaf, so the gate routes to `reclaim_or_refuse` → `FitInProgress`).
    let intruder_refused = Arc::new(AtomicBool::new(false));

    // RAII disarm: restore the hook to `None` on every exit path (incl. an
    // assertion unwind), so it can never leak into another test.
    struct DisarmOnDrop;
    impl Drop for DisarmOnDrop {
        fn drop(&mut self) {
            *super::CLEAR_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    let _disarm = DisarmOnDrop;

    {
        let store_h = Arc::clone(&store);
        let leaf_h = leaf.clone();
        let fired_h = Arc::clone(&fired);
        let refused_h = Arc::clone(&intruder_refused);
        let mut guard = super::CLEAR_GAP_HOOK.lock().unwrap();
        *guard = Some(Arc::new(move |_dir: &std::path::Path| {
            if fired_h.swap(true, Ordering::SeqCst) {
                return; // fire only on the first claim's clear window
            }
            let leaf2 = leaf_h.clone();
            let store2 = Arc::clone(&store_h);
            let refused2 = Arc::clone(&refused_h);
            std::thread::spawn(move || {
                match store2.claim_streaming(&leaf2, record(id(0xaa))) {
                    // The buggy code quarantines the held leaf and the intruder
                    // wins a `Fresh` claim — which the fix forbids.
                    Ok(claim) => {
                        claim
                            .write("chain_intruder/trace.tsv", b"sweep\tll\n1\t-1.0\n")
                            .unwrap();
                        claim.finalize(record(id(0xaa))).unwrap();
                    }
                    // Correct: a live `.lock` holds the leaf → refused.
                    Err(CasError::FitInProgress { .. }) | Err(CasError::AlreadyCompleted { .. }) => {
                        refused2.store(true, Ordering::SeqCst);
                    }
                    Err(e) => panic!("intruder hit an unexpected error: {e:?}"),
                }
            })
            .join()
            .unwrap();
        }));
    }

    // The reclaiming winner runs: reclaims the dead lock, clears the leaf
    // (firing the hook with the intruder), then writes `Running`. On the buggy
    // code its dir is quarantined mid-write → the write fails `Io(NotFound)`.
    let winner = store.claim_streaming(&leaf, record(id(0xaa)));

    *super::CLEAR_GAP_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;

    let claim = winner.expect("winner's claim must succeed — its leaf was quarantined out from under it (Io NotFound)");
    claim
        .write("chain_winner/trace.tsv", b"sweep\tll\n1\t-2.0\n")
        .unwrap();
    claim.finalize(record(id(0xaa))).unwrap();

    assert!(fired.load(Ordering::SeqCst), "the clear-gap hook never fired — the window was not exercised");
    assert!(
        intruder_refused.load(Ordering::SeqCst),
        "the intruder was NOT refused — it quarantined a leaf held by a live `.lock` (the false-orphan race is open)"
    );
    // The winner's leaf must be a single intact Completed Hit, not clobbered.
    assert!(
        matches!(store.lookup(&leaf, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)),
        "the winner's Completed leaf must be intact, not clobbered/quarantined"
    );
    cleanup(&root);
}

/// Deterministic proof of the `.reclaim`-recycle double-reclaim race in
/// `reclaim_or_refuse` (the residual that `mode_b_reclaim_is_exclusive_under_
/// concurrency` only catches probabilistically, and only on fast filesystems).
///
/// `reclaim_or_refuse` reads the holder PID once, dead-checks it, then acquires
/// `.reclaim`. The `.reclaim` O_EXCL gate is released at the END of the function
/// — *before* the critical section — so it is recyclable. A losing reclaimer
/// that read the dead PID, then stalled before `create_new(.reclaim)`, can — by
/// the time it owns the recycled `.reclaim` — be looking at a lock a concurrent
/// winner already took over (live PID). Without re-confirming the holder under
/// `.reclaim`, the loser renames its own PID over the winner's live `.lock` and
/// enters the critical section too: TWO claimants clearing + finalizing the
/// same leaf (`max_cs == 2`), a silent double-write of a fit result.
///
/// The `RECLAIM_PREACQUIRE_HOOK` parks the loser at exactly that point and
/// drives a winner through a full takeover first; the loser must then be
/// refused (re-confirm catches the now-live lock), not double-reclaim.
#[test]
fn mode_b_reclaim_recycle_is_exclusive() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let root = tmp_root("reclaim_recycle");
    let store = Arc::new(FsCasStore::new(&root));
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");

    // A definitely-dead PID for the planted lock (so a reclaim is warranted).
    let mut child = std::process::Command::new("true").spawn().expect("spawn true");
    let dead_pid = child.id();
    child.wait().expect("reap");

    // Plant a crashed run: a `Running` leaf + orphan whose `.lock` holder is dead.
    fs::remove_dir_all(&leaf).ok();
    let c = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    c.write("orphan_chain.tsv", b"partial").unwrap();
    drop(c);
    fs::write(leaf.join(".lock"), dead_pid.to_string()).unwrap();

    let fired = Arc::new(AtomicBool::new(false));
    let winner_done = Arc::new(AtomicBool::new(false));
    // Peak concurrent occupants of the critical section. Exclusivity ⇒ 1.
    let max_cs = Arc::new(AtomicUsize::new(0));

    // RAII disarm: clear the process-global hook on every exit path.
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            *super::RECLAIM_PREACQUIRE_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
    let _disarm = Disarm;

    {
        let store_h = Arc::clone(&store);
        let leaf_h = leaf.clone();
        let fired_h = Arc::clone(&fired);
        let winner_done_h = Arc::clone(&winner_done);
        let max_cs_h = Arc::clone(&max_cs);
        let mut guard = super::RECLAIM_PREACQUIRE_HOOK.lock().unwrap();
        *guard = Some(Arc::new(move |_lock: &std::path::Path| {
            // Fire ONCE — on the loser's reclaim, not the winner's re-entrant one.
            if fired_h.swap(true, Ordering::SeqCst) {
                return;
            }
            // Drive the winner through a FULL takeover (live PID over `.lock`,
            // releases `.reclaim`) and leave it in the critical section — Running,
            // NOT finalized — so the leaf stays clobberable. StreamClaim has no
            // Drop, so the on-disk live `.lock` + Running record persist after
            // the thread exits.
            let store2 = Arc::clone(&store_h);
            let leaf2 = leaf_h.clone();
            let max_cs2 = Arc::clone(&max_cs_h);
            let winner_done2 = Arc::clone(&winner_done_h);
            std::thread::spawn(move || {
                let claim = store2
                    .claim_streaming(&leaf2, record(id(0xaa)))
                    .expect("winner must reclaim the dead lock");
                claim.write("chain_winner/trace.tsv", b"sweep\tll\n1\t-1.0\n").unwrap();
                max_cs2.fetch_max(1, Ordering::SeqCst); // winner occupies the CS
                winner_done2.store(true, Ordering::SeqCst);
                std::mem::forget(claim); // keep the live `.lock` (no Drop, but be explicit)
            })
            .join()
            .unwrap();
        }));
    }

    // MAIN thread is the LOSER: resolve → Reclaim → create_new(.lock) fails (dead
    // lock present) → reclaim_or_refuse → dead-check passes → [hook parks it while
    // the winner takes over] → resume. With the re-confirm it must be REFUSED;
    // without it, it recycles `.reclaim`, takes over the winner's live lock, and
    // enters the CS too.
    let loser = store.claim_streaming(&leaf, record(id(0xaa)));
    *super::RECLAIM_PREACQUIRE_HOOK.lock().unwrap_or_else(|e| e.into_inner()) = None;

    assert!(fired.load(Ordering::SeqCst), "pre-acquire hook never fired — window not exercised");
    assert!(winner_done.load(Ordering::SeqCst), "winner never reached the critical section");

    match loser {
        Err(CasError::FitInProgress { .. }) | Err(CasError::AlreadyCompleted { .. }) => {}
        Ok(_) => {
            max_cs.fetch_max(2, Ordering::SeqCst); // loser illegally entered too
            panic!(
                "the loser was NOT refused — it recycled `.reclaim` and took over \
                 a live `.lock` (double-reclaim: two claimants in the critical \
                 section, max_cs=2)"
            );
        }
        Err(e) => panic!("loser hit an unexpected error: {e:?}"),
    }
    assert_eq!(
        max_cs.load(Ordering::SeqCst),
        1,
        "exactly one claimant may hold the critical section",
    );
    cleanup(&root);
}

#[test]
fn mode_b_reclaim_is_exclusive_under_concurrency() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Serialize against the gap-hook test: an installed hook is a process-global.
    let _serial = super::RECLAIM_HOOK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = tmp_root("reclaim_race");
    let store = Arc::new(FsCasStore::new(&root));
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");

    // A definitely-dead PID (a child we spawn and reap) for the planted lock.
    let mut child = std::process::Command::new("true").spawn().expect("spawn true");
    let dead_pid = child.id();
    child.wait().expect("reap");

    // (Re)plant a crashed run: a `Running` leaf + orphan whose `.lock` holder
    // PID is dead — the state a fit killed before `finalize` leaves behind.
    let plant = || {
        fs::remove_dir_all(&leaf).ok();
        let c = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
        c.write("orphan_chain.tsv", b"partial").unwrap();
        drop(c); // no finalize → `.lock` + Running linger; overwrite with a dead PID
        fs::write(leaf.join(".lock"), dead_pid.to_string()).unwrap();
    };

    // N processes re-launch the SAME killed fit at once (e.g. a cluster
    // re-submit). The dead lock must be reclaimed by EXACTLY ONE — never two
    // both removing it and both writing the shared chain files. Several rounds
    // raise the odds a racy reclaim is caught; the serialized reclaim makes
    // `max_cs <= 1` invariant, so a correct build is green every round.
    const ROUNDS: usize = 8;
    const N: usize = 12;
    for round in 0..ROUNDS {
        plant();
        let in_cs = Arc::new(AtomicUsize::new(0));
        let max_cs = Arc::new(AtomicUsize::new(0));
        let successes = Arc::new(AtomicUsize::new(0));
        let resolved = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..N {
            let store = Arc::clone(&store);
            let leaf = leaf.clone();
            let (in_cs, max_cs) = (Arc::clone(&in_cs), Arc::clone(&max_cs));
            let (successes, resolved) = (Arc::clone(&successes), Arc::clone(&resolved));
            handles.push(std::thread::spawn(move || {
                match store.claim_streaming(&leaf, record(id(0xaa))) {
                    Ok(claim) => {
                        let now = in_cs.fetch_add(1, Ordering::SeqCst) + 1;
                        max_cs.fetch_max(now, Ordering::SeqCst);
                        // Widen the critical section so any intruder is observed.
                        claim.write("chain_1/trace.tsv", b"sweep\tll\n1\t-1.0\n").unwrap();
                        std::thread::sleep(std::time::Duration::from_millis(8));
                        in_cs.fetch_sub(1, Ordering::SeqCst);
                        claim.finalize(record(id(0xaa))).unwrap();
                        successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(CasError::FitInProgress { .. }) | Err(CasError::AlreadyCompleted { .. }) => {}
                    Err(e) => panic!("unexpected claim error: {e:?}"),
                }
                resolved.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            max_cs.load(Ordering::SeqCst),
            1,
            "round {round}: two claimants entered the critical section (double-write)"
        );
        assert_eq!(
            successes.load(Ordering::SeqCst),
            1,
            "round {round}: exactly one re-launch must win the reclaim"
        );
        assert_eq!(resolved.load(Ordering::SeqCst), N, "round {round}: every claimant resolved");
        assert!(
            !leaf.join("orphan_chain.tsv").exists(),
            "round {round}: the crashed orphan was cleared by the winner"
        );
        assert!(matches!(
            store.lookup(&leaf, &LeafIdentity::new(id(0xaa))),
            Lookup::Hit(_)
        ));
    }
    cleanup(&root);
}

// ── Mode B: recursive (nested) exact-set manifest — fit stages nest chains ───

/// Build a record carrying declared child namespaces (e.g. `trajectories`).
fn record_with_children(run_id: ContentHash, children: &[(&str, ContentHash)]) -> RunRecord {
    let mut r = record(run_id);
    for (ns, cid) in children {
        r.children.insert((*ns).to_string(), vec![*cid]);
    }
    r
}

#[test]
fn mode_b_nested_own_files_are_manifested_and_hit() {
    // A fit stage streams per-chain files under `chain_N/` plus stage-root
    // outputs. The recursive manifest must capture every nested own-file
    // (keyed by its `/`-joined relative path), and lookup must Hit.
    let root = tmp_root("nested_hit");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb").join("seed_1-cccccccc");
    let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
    claim.write("chain_1/trace.tsv", b"sweep\tll\n1\t-2.0\n").unwrap();
    claim.write("chain_2/trace.tsv", b"sweep\tll\n1\t-2.1\n").unwrap();
    claim.write("fit_state.toml", b"best_loglik = -2.0\n").unwrap();
    let dest = claim.finalize(record(id(0xaa))).unwrap();

    match store.lookup(&dest, &LeafIdentity::new(id(0xaa))) {
        Lookup::Hit(r) => {
            assert!(r.artifacts.contains_key("chain_1/trace.tsv"), "nested chain file manifested");
            assert!(r.artifacts.contains_key("chain_2/trace.tsv"));
            assert!(r.artifacts.contains_key("fit_state.toml"));
            assert!(!r.artifacts.contains_key("chain_1"), "the dir itself is not an own-file");
        }
        other => panic!("expected Hit, got {other:?}"),
    }
    cleanup(&root);
}

#[test]
fn mode_b_nested_missing_file_is_corrupt() {
    let root = tmp_root("nested_missing");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("f-aaaaaaaa").join("01-s-bbbbbbbb").join("seed_1-cccccccc");
    let dest = {
        let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
        claim.write("chain_1/trace.tsv", b"data\n").unwrap();
        claim.write("fit_state.toml", b"x = 1\n").unwrap();
        claim.finalize(record(id(0xaa))).unwrap()
    };
    fs::remove_file(dest.join("chain_1/trace.tsv")).unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::Corrupt)
    ), "a missing nested own-file is Corrupt, not a Hit");
    cleanup(&root);
}

#[test]
fn mode_b_nested_orphan_file_is_orphan() {
    let root = tmp_root("nested_orphan");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("f-aaaaaaaa").join("01-s-bbbbbbbb").join("seed_1-cccccccc");
    let dest = {
        let claim = store.claim_streaming(&leaf, record(id(0xaa))).unwrap();
        claim.write("chain_1/trace.tsv", b"data\n").unwrap();
        claim.finalize(record(id(0xaa))).unwrap()
    };
    // An unlisted file dropped into a nested own-dir (crash debris) is an orphan.
    fs::write(dest.join("chain_1/extra.tsv"), b"debris").unwrap();
    assert!(matches!(
        store.lookup(&dest, &LeafIdentity::new(id(0xaa))),
        Lookup::Stale(StaleReason::OrphanFiles)
    ), "an unlisted nested file is an orphan");
    cleanup(&root);
}

#[test]
fn mode_b_declared_child_is_a_boundary_not_recursed() {
    // `trajectories/` is a declared child (its own artifact, keyed on
    // n_trajectories). Its files must NOT enter the stage manifest and must
    // NOT be orphaned — the recursive walk stops at the child boundary.
    let root = tmp_root("child_boundary");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("f-aaaaaaaa").join("01-s-bbbbbbbb").join("seed_1-cccccccc");
    let rec = record_with_children(id(0xaa), &[("trajectories", id(0xcc))]);
    let claim = store.claim_streaming(&leaf, rec.clone()).unwrap();
    claim.write("fit_state.toml", b"x = 1\n").unwrap();
    // Files inside the declared child — written as the child's content.
    claim.write("trajectories/000001.tsv", b"t\tS\n0\t99\n").unwrap();
    claim.write("trajectories/000002.tsv", b"t\tS\n0\t98\n").unwrap();
    let dest = claim.finalize(rec).unwrap();

    match store.lookup(&dest, &LeafIdentity::new(id(0xaa))) {
        Lookup::Hit(r) => {
            assert!(r.artifacts.contains_key("fit_state.toml"));
            // The child's files are NOT folded into the stage manifest — so
            // changing n_trajectories cannot re-key the θ̂ leaf.
            assert!(!r.artifacts.keys().any(|k| k.starts_with("trajectories/")),
                "declared-child files must not enter the stage manifest: {:?}", r.artifacts.keys().collect::<Vec<_>>());
        }
        other => panic!("expected Hit (child is a recognized boundary), got {other:?}"),
    }
    cleanup(&root);
}

#[test]
fn mode_b_collision_disambiguates() {
    let root = tmp_root("modebcoll");
    let store = FsCasStore::new(&root);
    let leaf = root.join("fits").join("fit-aaaaaaaa").join("01-scout-bbbbbbbb");
    // A completed leaf of identity A occupies the path.
    store
        .claim_streaming(&leaf, record(id(0xaa)))
        .unwrap()
        .finalize(record(id(0xaa)))
        .unwrap();
    // A different identity B claiming the same path disambiguates, never
    // clobbering A.
    let claim_b = store.claim_streaming(&leaf, record(id(0xbb))).unwrap();
    assert_ne!(claim_b.dir(), leaf.as_path());
    assert!(claim_b.dir().file_name().unwrap().to_string_lossy().contains('~'));
    // A is untouched.
    assert!(matches!(store.lookup(&leaf, &LeafIdentity::new(id(0xaa))), Lookup::Hit(_)));
    cleanup(&root);
}
