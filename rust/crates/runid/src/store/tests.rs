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
        ir_version: 7,
        engine_version: "0.3.0".into(),
        levels: vec![],
        deps: vec![],
        status: RunStatus::Running,
        artifacts: BTreeMap::new(),
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
    // A second identical-identity commit finds the completed leaf → benign
    // race, same destination, incumbent bytes preserved.
    let p2 = store.commit_atomic(&leaf, record(id(0xaa)), arts(b"SHOULD_NOT_WIN")).unwrap();
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
