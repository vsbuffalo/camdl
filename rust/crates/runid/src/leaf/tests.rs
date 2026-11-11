//! `leaf` tests: each way a candidate is unreadable, each leaf state the
//! reader reports, the interrupted-run guard, what discovery walks and in
//! which order, and the work a scoped walk avoids.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::*;
use crate::hash::HASH_VERSION;
use crate::record::{Failure, LevelId, Provenance, RunStatus, FORMAT_VERSION};
use crate::store::FsCasStore;

fn nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
}

/// A fresh temp root, removed when dropped.
struct TmpRoot(PathBuf);

impl TmpRoot {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir()
            .join(format!("runid_leaf_{}_{}_{}", tag, std::process::id(), nanos()));
        fs::create_dir_all(&p).unwrap();
        TmpRoot(p)
    }
}

impl std::ops::Deref for TmpRoot {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

fn id(byte: u8) -> ContentHash {
    ContentHash::from_bytes([byte; 32])
}

/// A 64-hex `run_id` from a short readable stem.
fn hex_id(stem: &str) -> ContentHash {
    ContentHash::from_hex(&format!("{:0<64}", stem)).unwrap()
}

fn record(kind: ArtifactKind, run_id: ContentHash, status: RunStatus) -> RunRecord {
    RunRecord {
        format_version: FORMAT_VERSION,
        kind,
        run_id,
        hash_version: HASH_VERSION,
        ir_version: "0.7".into(),
        engine_version: "0.3.0".into(),
        levels: vec![LevelId {
            name: "seed".into(),
            label: "seed_1".into(),
            hash: run_id,
            schema_version: 1,
        }],
        deps: vec![],
        status,
        failure: None,
        artifacts: BTreeMap::new(),
        output_schema: BTreeMap::new(),
        children: BTreeMap::new(),
        inputs: serde_json::Value::Null,
        provenance: Provenance::default(),
    }
}

/// Write `rec` as `dir/run.json`, creating `dir`. Returns `dir`.
fn plant(dir: &Path, rec: &RunRecord) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("run.json"), serde_json::to_vec_pretty(rec).unwrap()).unwrap();
    dir.to_path_buf()
}

/// Write raw bytes as `dir/run.json`, creating `dir`. Returns `dir`.
fn plant_bytes(dir: &Path, bytes: &[u8]) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("run.json"), bytes).unwrap();
    dir.to_path_buf()
}

fn unreadable(dir: &Path) -> UnreadableLeaf {
    match read_leaf(dir) {
        Err(u) => u,
        other => panic!("expected an unreadable leaf at {}, got {other:?}", dir.display()),
    }
}

fn state(dir: &Path) -> LeafState {
    match read_leaf(dir) {
        Ok(Some(leaf)) => leaf.state,
        other => panic!("expected a leaf at {}, got {other:?}", dir.display()),
    }
}

/// The pid of a process that has exited and been reaped.
fn dead_pid() -> u32 {
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    pid
}

fn dirs(entries: &[Result<Leaf, UnreadableLeaf>]) -> Vec<PathBuf> {
    entries.iter().map(|e| entry_dir(e).to_path_buf()).collect()
}

// ── not a leaf ───────────────────────────────────────────────────────────────

#[test]
fn a_directory_with_neither_record_nor_lock_is_not_a_leaf() {
    let root = TmpRoot::new("empty");
    assert!(matches!(read_leaf(&root), Ok(None)));
    assert!(matches!(read_leaf(&root.join("absent")), Ok(None)));
    assert!(discover(&root, Scope::default()).is_empty());
}

// ── each way a candidate is unreadable ───────────────────────────────────────

#[test]
fn garbage_bytes_are_malformed_with_no_header() {
    let root = TmpRoot::new("garbage");
    let dir = plant_bytes(&root.join("sims/a"), b"\x00\xffnot json at all");
    let u = unreadable(&dir);
    assert!(matches!(u.reason, Unreadable::Malformed { .. }), "{u:?}");
    assert_eq!((u.kind, u.run_id), (None, None));
    assert_eq!(u.dir, dir);
}

#[test]
fn truncated_json_is_malformed_and_keeps_its_header() {
    let root = TmpRoot::new("truncated");
    let full = serde_json::to_vec_pretty(&record(ArtifactKind::Sim, id(7), RunStatus::Completed))
        .unwrap();
    let dir = plant_bytes(&root.join("sims/a"), &full[..full.len() / 2]);
    let u = unreadable(&dir);
    assert!(matches!(u.reason, Unreadable::Malformed { .. }), "{u:?}");
    assert_eq!(u.kind, Some(ArtifactKind::Sim), "the header precedes the cut");
    assert_eq!(u.run_id, Some(id(7)));
}

#[test]
fn a_renamed_kind_is_named_with_the_sentence_readers_already_print() {
    let root = TmpRoot::new("renamed");
    let mut v =
        serde_json::to_value(record(ArtifactKind::FitMethod, id(3), RunStatus::Completed)).unwrap();
    let (old, new) = crate::record::RENAMED_KINDS[0];
    v["kind"] = old.into();
    let bytes = serde_json::to_vec(&v).unwrap();
    let dir = plant_bytes(&root.join("fits/a"), &bytes);
    let u = unreadable(&dir);
    assert_eq!(u.reason, Unreadable::RenamedKind { old: old.into(), new: new.into() });
    assert_eq!(u.reason.to_string(), crate::record::renamed_kind(&bytes).unwrap());
    // Its kind is the unknown one, so even a kind-scoped walk keeps it.
    let found = discover_kind(&root, ArtifactKind::FitMethod, Scope::default());
    assert!(matches!(found.as_slice(), [Err(u)] if u.dir == dir), "{found:?}");
}

#[test]
#[cfg(unix)]
fn a_run_json_this_process_may_not_read_is_an_io_entry() {
    use std::os::unix::fs::PermissionsExt;
    let root = TmpRoot::new("mode000");
    let dir = plant(&root.join("sims/a"), &record(ArtifactKind::Sim, id(1), RunStatus::Completed));
    let rj = dir.join("run.json");
    fs::set_permissions(&rj, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::read(&rj).is_ok() {
        // Running as a user the mode does not bind (root): nothing to test.
        fs::set_permissions(&rj, fs::Permissions::from_mode(0o644)).unwrap();
        return;
    }
    let u = unreadable(&dir);
    assert!(
        matches!(u.reason, Unreadable::Io { kind: ErrorKind::PermissionDenied, .. }),
        "{u:?}"
    );
    let found = discover_kind(&root, ArtifactKind::Sim, Scope::default());
    assert!(matches!(found.as_slice(), [Err(u)] if u.dir == dir), "{found:?}");
    fs::set_permissions(&rj, fs::Permissions::from_mode(0o644)).unwrap();
}

/// A directory the walk cannot list or look into may hold a leaf; it is an
/// entry, not a silent gap in the walk.
#[test]
#[cfg(unix)]
fn a_directory_this_process_may_not_enter_is_an_io_entry() {
    use std::os::unix::fs::PermissionsExt;
    let root = TmpRoot::new("dir000");
    let dir = plant(&root.join("sims/a"), &record(ArtifactKind::Sim, id(1), RunStatus::Completed));
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o000)).unwrap();
    let enforced = fs::metadata(dir.join("run.json")).is_err();
    let found = discover(&root, Scope::default());
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    if !enforced {
        return; // Running as a user the mode does not bind (root).
    }
    assert!(
        matches!(found.as_slice(), [Err(UnreadableLeaf { reason: Unreadable::Io { kind: ErrorKind::PermissionDenied, .. }, .. })]),
        "{found:?}"
    );
}

#[test]
fn a_directory_with_a_lock_and_no_record_is_no_record_with_its_owner() {
    let root = TmpRoot::new("lockonly");
    let live = root.join("fits/live");
    fs::create_dir_all(&live).unwrap();
    fs::write(live.join(".lock"), format!("{}\n", std::process::id())).unwrap();
    let pid = std::process::id();
    assert_eq!(
        unreadable(&live).reason,
        Unreadable::NoRecord { owner: LockOwner::Live { pid } }
    );

    let debris = root.join("fits/debris");
    fs::create_dir_all(&debris).unwrap();
    let dead = dead_pid();
    fs::write(debris.join(".lock"), format!("{dead}\n")).unwrap();
    assert_eq!(
        unreadable(&debris).reason,
        Unreadable::NoRecord { owner: LockOwner::Dead { pid: dead } }
    );

    let found = discover_kind(&root, ArtifactKind::FitMethod, Scope::default());
    assert_eq!(dirs(&found), vec![debris.clone(), live.clone()]);
    assert!(found.iter().all(|e| matches!(e, Err(UnreadableLeaf { reason: Unreadable::NoRecord { .. }, .. }))));
}

// ── leaf states ──────────────────────────────────────────────────────────────

#[test]
fn a_failed_leaf_carries_its_recorded_failure_or_says_it_has_none() {
    let root = TmpRoot::new("failed");
    let store = FsCasStore::new(&*root);
    let with = root.join("fits/with");
    let claim = store.claim_streaming(&with, record(ArtifactKind::FitMethod, id(1), RunStatus::Running)).unwrap();
    let failure = Failure { message: "the likelihood is -inf".into(), operation: "pmmh".into() };
    claim.fail(failure.clone()).unwrap();
    assert_eq!(state(&with), LeafState::Failed { failure });

    let without = root.join("fits/without");
    drop(store.claim_streaming(&without, record(ArtifactKind::FitMethod, id(2), RunStatus::Running)).unwrap());
    assert_eq!(state(&without), LeafState::FailedReasonUnknown);
}

#[test]
fn a_running_leaf_whose_owner_is_gone_is_interrupted() {
    let root = TmpRoot::new("interrupted");
    let dead = dead_pid();
    let killed = plant(&root.join("fits/killed"), &record(ArtifactKind::FitMethod, id(1), RunStatus::Running));
    fs::write(killed.join(".lock"), format!("{dead}\n")).unwrap();
    assert_eq!(state(&killed), LeafState::Interrupted { pid: Some(dead) });

    let released = plant(&root.join("fits/released"), &record(ArtifactKind::FitMethod, id(2), RunStatus::Running));
    assert_eq!(state(&released), LeafState::Interrupted { pid: None });
}

#[test]
fn a_running_leaf_held_by_a_live_claim_is_running() {
    let root = TmpRoot::new("live");
    let store = FsCasStore::new(&*root);
    let dir = root.join("fits/live");
    let claim = store.claim_streaming(&dir, record(ArtifactKind::FitMethod, id(1), RunStatus::Running)).unwrap();
    assert_eq!(state(&dir), LeafState::Running { pid: Some(std::process::id()) });
    claim.finalize(record(ArtifactKind::FitMethod, id(1), RunStatus::Running)).unwrap();
    assert_eq!(state(&dir), LeafState::Completed);
}

#[test]
fn a_running_leaf_locked_on_another_host_is_owner_unknown() {
    let root = TmpRoot::new("foreign");
    let dir = plant(&root.join("fits/remote"), &record(ArtifactKind::FitMethod, id(1), RunStatus::Running));
    let host = "a-host-that-is-not-this-one.invalid";
    fs::write(dir.join(".lock"), format!("4242\n{host}\n")).unwrap();
    assert_eq!(state(&dir), LeafState::OwnerUnknown { pid: 4242, host: host.into() });
}

/// A writer that commits `Completed` and releases its `.lock` between the
/// reader's record read and its lock read must not be reported interrupted.
#[test]
fn a_run_finalized_between_the_reads_is_not_reported_interrupted() {
    let root = TmpRoot::new("snapshot");
    let dir = plant(&root.join("fits/racing"), &record(ArtifactKind::FitMethod, id(1), RunStatus::Running));
    let completed = record(ArtifactKind::FitMethod, id(1), RunStatus::Completed);
    SNAPSHOT_GAP_HOOK.with(|h| {
        *h.borrow_mut() = Some(Box::new(move |d: &Path| {
            plant(d, &completed);
        }))
    });
    let got = state(&dir);
    SNAPSHOT_GAP_HOOK.with(|h| *h.borrow_mut() = None);
    assert_eq!(got, LeafState::Completed);
    // With no writer in the gap the verdict stands.
    let still = plant(&root.join("fits/still"), &record(ArtifactKind::FitMethod, id(2), RunStatus::Running));
    assert_eq!(state(&still), LeafState::Interrupted { pid: None });
}

// ── what discovery walks ─────────────────────────────────────────────────────

/// A retried leaf keeps its earlier attempts under `.attempts/`. None of them
/// is a second leaf of the same identity — not the real retry layout, whose
/// record is renamed `attempt.json`, and not a `run.json` planted there.
#[test]
fn preserved_attempts_are_never_discovered() {
    let root = TmpRoot::new("attempts");
    let store = FsCasStore::new(&*root);
    let leaf = root.join("fits/f-aaaaaaaa/seed_1-bbbbbbbb");
    let rec = || record(ArtifactKind::FitMethod, id(0xaa), RunStatus::Running);
    store
        .claim_streaming(&leaf, rec())
        .unwrap()
        .fail(Failure { message: "first".into(), operation: "pmmh".into() })
        .unwrap();
    let claim = store.claim_streaming(&leaf, rec()).unwrap();
    claim.write("traj.tsv", b"S\tI\n9\t1\n").unwrap();
    claim.finalize(rec()).unwrap();
    assert!(leaf.join(ATTEMPTS_DIR).join("1").is_dir(), "the retry preserved an attempt");
    plant(&leaf.join(ATTEMPTS_DIR).join("9"), &rec());

    for found in [
        discover(&root, Scope::default()),
        discover_kind(&root, ArtifactKind::FitMethod, Scope::default()),
        discover(&leaf, Scope::default()),
    ] {
        assert_eq!(dirs(&found), vec![leaf.clone()]);
        assert!(matches!(&found[0], Ok(l) if l.state == LeafState::Completed));
    }
}

#[test]
fn discovery_returns_path_order() {
    let root = TmpRoot::new("order");
    let names = ["b/2", "a", "c/1/x", "b/10", "b/1", "c"];
    for (i, n) in names.iter().enumerate() {
        plant(&root.join("sims").join(n), &record(ArtifactKind::Sim, id(i as u8), RunStatus::Completed));
    }
    plant_bytes(&root.join("sims/b/0"), b"garbage");
    let mut want: Vec<PathBuf> = names.iter().map(|n| root.join("sims").join(n)).collect();
    want.push(root.join("sims/b/0"));
    want.sort();
    assert_eq!(dirs(&discover(&root, Scope::default())), want);
    assert_eq!(dirs(&discover_kind(&root, ArtifactKind::Sim, Scope::default())), want);
}

/// A scope excludes an unreadable record only on its header's evidence.
#[test]
fn a_scope_excludes_an_unreadable_record_only_on_its_header() {
    let root = TmpRoot::new("scope");
    let sims = root.join("sims");
    let cut = |kind, run_id| {
        let b = serde_json::to_vec_pretty(&record(kind, run_id, RunStatus::Completed)).unwrap();
        b[..b.len() / 2].to_vec()
    };
    let garbage = plant_bytes(&sims.join("garbage"), b"{");
    let other_kind = plant_bytes(&sims.join("pfilter"), &cut(ArtifactKind::Pfilter, id(1)));
    let skipped = plant_bytes(&sims.join("skipped"), &cut(ArtifactKind::Sim, id(2)));
    let kept = plant_bytes(&sims.join("kept"), &cut(ArtifactKind::Sim, id(3)));

    let skip: HashSet<ContentHash> = [id(2)].into();
    let found = discover_kind(&root, ArtifactKind::Sim, Scope { skip: Some(&skip), ..Scope::default() });
    let mut want = vec![garbage.clone(), kept.clone()];
    want.sort();
    assert_eq!(dirs(&found), want);
    assert!(found.iter().all(|e| e.is_err()));

    // Unscoped, every candidate is an entry.
    let mut all = vec![garbage, other_kind, skipped, kept];
    all.sort();
    assert_eq!(dirs(&discover(&sims, Scope::default())), all);
}

/// The walk follows symlinks, both to a subtree and to a `run.json`.
/// Reading entry types out of the directory listing (`DirEntry::file_type`)
/// instead of `stat`ing each path is what makes the walk cheap, but it
/// reports the *link*, not the target — so a symlinked subtree would
/// silently stop being walked, and a symlinked `run.json` would stop being
/// a leaf. Both are still resolved.
#[test]
#[cfg(unix)]
fn symlinked_subtrees_and_records_are_still_walked() {
    let root = TmpRoot::new("symlinks");
    let elsewhere = root.join("elsewhere");
    let store = root.join("store");

    // A leaf reached only through a symlinked directory.
    plant(&elsewhere.join("sims/m-1111/cfg-2222"), &record(ArtifactKind::Sim, id(0xbe), RunStatus::Completed));
    fs::create_dir_all(store.join("sims")).unwrap();
    std::os::unix::fs::symlink(elsewhere.join("sims/m-1111"), store.join("sims/linked-3333")).unwrap();

    // A leaf whose `run.json` is itself a symlink to a record elsewhere.
    let donor = plant(&elsewhere.join("sims/donor-4444/cfg-5555"), &record(ArtifactKind::Sim, id(0xca), RunStatus::Completed));
    let via_link = store.join("sims/via-link-6666");
    fs::create_dir_all(&via_link).unwrap();
    std::os::unix::fs::symlink(donor.join("run.json"), via_link.join("run.json")).unwrap();

    let mut got: Vec<ContentHash> = discover_kind(&store, ArtifactKind::Sim, Scope::default())
        .into_iter()
        .map(|e| e.unwrap().record.run_id)
        .collect();
    got.sort_by_key(|h| h.to_hex());
    assert_eq!(got, vec![id(0xbe), id(0xca)]);
}

// ── the cheap header and the work a scoped walk avoids ───────────────────────

/// Zero this thread's parse counter before the walk being measured.
fn measuring() {
    FULL_PARSES.with(|n| n.set(0));
}

fn full_parses() -> usize {
    FULL_PARSES.with(|n| n.get())
}

/// Plant a `Sim` leaf at `root/sims/<sub>/` whose record carries a
/// kilobyte-scale `inputs` payload, the shape a real per-draw leaf has (that
/// payload is exactly what the scoped walk must not pay to materialize).
fn plant_sim(root: &Path, sub: &str, run_id: ContentHash) -> PathBuf {
    let mut rec = record(ArtifactKind::Sim, run_id, RunStatus::Completed);
    let params: serde_json::Map<String, serde_json::Value> =
        (0..40).map(|i| (format!("theta_{i}"), (i as f64 + 0.5).into())).collect();
    rec.inputs = serde_json::json!({"backend": "chain_binomial", "dt": 1.0, "params": params});
    plant(&root.join("sims").join(sub), &rec)
}

/// A store shaped like the one in gh#699: most sim leaves are members of an
/// ensemble that already represents them. Returns (member ids, free ids).
fn ensemble_heavy_store(root: &Path, n_members: usize, n_free: usize) -> (Vec<ContentHash>, Vec<ContentHash>) {
    let members: Vec<ContentHash> = (0..n_members).map(|i| hex_id(&format!("aa{i:04x}"))).collect();
    for (i, m) in members.iter().enumerate() {
        plant_sim(root, &format!("m-1111/draw{i}-2222/baseline-3333/seed_1-4444"), *m);
    }
    let free: Vec<ContentHash> = (0..n_free).map(|i| hex_id(&format!("ff{i:04x}"))).collect();
    for (i, f) in free.iter().enumerate() {
        plant_sim(root, &format!("solo-5555/cfg{i}-6666/baseline-3333/seed_1-4444"), *f);
    }
    (members, free)
}

fn sorted_ids(entries: Vec<Result<Leaf, UnreadableLeaf>>) -> Vec<String> {
    let mut ids: Vec<String> = entries.into_iter().map(|e| e.unwrap().record.run_id.to_hex()).collect();
    ids.sort();
    ids
}

fn sorted_hex(ids: &[ContentHash]) -> Vec<String> {
    let mut v: Vec<String> = ids.iter().map(|h| h.to_hex()).collect();
    v.sort();
    v
}

/// gh#699. `camdl list` suppresses every sim leaf an ensemble row already
/// represents — so those leaves must never cost a full `RunRecord` parse.
/// The walk may only materialize the leaves that survive to be printed.
#[test]
fn skipped_members_are_excluded_without_a_full_parse() {
    let root = TmpRoot::new("skip");
    let (members, free) = ensemble_heavy_store(&root, 40, 5);
    let skip: HashSet<ContentHash> = members.iter().copied().collect();

    measuring();
    let found = discover_kind(&root, ArtifactKind::Sim, Scope { skip: Some(&skip), ..Scope::default() });
    let parses = full_parses();

    assert_eq!(sorted_ids(found), sorted_hex(&free), "exactly the non-member leaves");
    assert_eq!(
        parses,
        free.len(),
        "members must not be fully parsed ({} sim leaves in the tree)",
        members.len() + free.len()
    );
}

/// Backdate a directory's mtime by `age`, so the walk sees it as written then.
fn backdate(dir: &Path, age: Duration) {
    let when = SystemTime::now() - age;
    let times = fs::FileTimes::new().set_accessed(when).set_modified(when);
    fs::File::options().read(true).open(dir).unwrap().set_times(times).unwrap();
}

/// gh#699. `--since` bounds discovery, not just the printed rows: a leaf
/// directory's mtime is an upper bound on the `created_at` inside it, so a
/// leaf older than the cutoff must never be read or parsed.
#[test]
fn leaves_older_than_the_since_cutoff_are_never_read() {
    let root = TmpRoot::new("since");
    let old: Vec<ContentHash> = (0..12).map(|i| hex_id(&format!("01d{i:02}"))).collect();
    for (i, o) in old.iter().enumerate() {
        let dir = plant_sim(&root, &format!("m-1111/old{i}-2222"), *o);
        backdate(&dir, Duration::from_secs(30 * 24 * 3600));
    }
    let fresh: Vec<ContentHash> = (0..3).map(|i| hex_id(&format!("0fe{i:02}"))).collect();
    for (i, f) in fresh.iter().enumerate() {
        plant_sim(&root, &format!("m-1111/new{i}-3333"), *f);
    }

    let cutoff = SystemTime::now() - Duration::from_secs(3600);
    measuring();
    let found = discover_kind(&root, ArtifactKind::Sim, Scope { not_before: Some(cutoff), ..Scope::default() });
    let parses = full_parses();
    assert_eq!(sorted_ids(found), sorted_hex(&fresh), "only the leaves inside the window");
    assert_eq!(parses, fresh.len(), "leaves outside the --since window must not be read");
}

/// The mtime prune is one-sided. A store whose directory mtimes are newer
/// than the records inside them — every `cp -r`/`rsync` copy — must not lose
/// leaves: the walk declines to prune and the caller's `created_at` filter
/// does the real work.
#[test]
fn a_recently_copied_store_is_not_pruned() {
    let root = TmpRoot::new("copied");
    let (_, free) = ensemble_heavy_store(&root, 0, 4);
    let cutoff = SystemTime::now() - Duration::from_secs(3600);
    let found = discover_kind(&root, ArtifactKind::Sim, Scope { not_before: Some(cutoff), ..Scope::default() });
    assert_eq!(sorted_ids(found), sorted_hex(&free), "a fresh mtime is never a proof of age");
}

/// The unscoped walk returns every leaf, members included: exclusion is
/// opt-in, never a property of the store.
#[test]
fn the_unscoped_walk_returns_every_leaf() {
    let root = TmpRoot::new("unscoped");
    let (members, free) = ensemble_heavy_store(&root, 6, 2);
    let all: Vec<ContentHash> = members.into_iter().chain(free).collect();
    assert_eq!(sorted_ids(discover_kind(&root, ArtifactKind::Sim, Scope::default())), sorted_hex(&all));
}

/// The header read stops at `run_id`: a record whose bytes after `run_id` are
/// unparseable still yields a header, while the full `RunRecord` parse of the
/// same bytes fails. That gap is what the walk is buying — a full
/// tokenization would fail here too, so this test fails if the early exit is
/// lost.
#[test]
fn run_header_stops_at_run_id() {
    let bytes = format!(
        r#"{{"format_version":1,"kind":"sim","run_id":"{}","levels":[ NOT JSON"#,
        hex_id("abcd").to_hex()
    );
    let hdr = run_header(bytes.as_bytes()).expect("header reads before the garbage");
    assert_eq!(hdr.kind, ArtifactKind::Sim);
    assert_eq!(hdr.run_id, hex_id("abcd"));
    assert!(serde_json::from_slice::<RunRecord>(bytes.as_bytes()).is_err());
}

/// Anything the header read rejects, the full parse rejects too — the
/// property that lets a scoped walk classify a header-less record as
/// unreadable without a full parse.
#[test]
fn a_record_without_identity_is_rejected_by_both_reads() {
    let bytes = br#"{"format_version":1,"kind":"sim","status":"completed"}"#;
    assert!(run_header(bytes).is_none());
    assert!(serde_json::from_slice::<RunRecord>(bytes).is_err());
}
