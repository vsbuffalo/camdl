//! `verify`, `clean`, `reindex`, the commands, the lock protocol and timestamps.

use std::sync::{Arc, Barrier};

use crate::tests::*;
use crate::fsutil::{try_lock, LockAttempt};
use crate::{Begin, Reuse, StoreCommand};

#[test]
fn verify_finds_a_file_whose_bytes_changed_behind_its_size_and_time() {
    let root = TempRoot::new("verify");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    assert!(store.verify(None).unwrap().is_empty());
    let path = s.folder().join("traj.tsv");
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::write(&path, b"t\tI\n0\t9\n").unwrap();
    std::fs::File::options().write(true).open(&path).unwrap().set_modified(mtime).unwrap();
    assert!(store.get(s.id()).unwrap().is_some(), "the cheap check sees size and time only");
    let failures = store.verify(None).unwrap();
    assert_eq!(failures.len(), 1);
    assert!(failures[0].what.contains("traj.tsv"));
    assert!(matches!(s.open("traj.tsv"), Err(crate::Error::Damaged { .. })), "a read checks the digest");
}

#[test]
fn verify_computes_the_run_id_again_from_the_identity_file() {
    let root = TempRoot::new("verify-id");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    let p = s.folder().join("identity.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    v["identity"]["seed"] = 2.into();
    std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
    let f = store.verify(None).unwrap();
    assert_eq!(f.len(), 1);
    assert!(f[0].what.contains("run id"), "{}", f[0].what);
}

#[test]
fn clean_removes_quarantine_attempts_dead_staging_and_empty_directories_only() {
    let root = TempRoot::new("clean");
    let store = open(root.path());
    let kept = run_sim(&store, &info(), 1);
    let damaged = run_sim(&store, &info(), 2);
    std::fs::write(damaged.folder().join("stray"), b"x").unwrap();
    run_sim(&store, &info(), 2);
    let f = FitId { problem: "p".into(), method: "m".into() };
    let Begin::Claimed(run) = store.begin::<Fit>(&f, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.fail("x", "y").unwrap();
    let Begin::Claimed(run) = store.begin::<Fit>(&f, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let fit = run.finish().unwrap();
    let (id, labels) = sim(3);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let staging = run.core.files_dir_pub();
    std::mem::forget(run);
    set_owner(&staging.join("run.json"), &dead_owner());
    let empty_group = kept.folder().parent().unwrap().parent().unwrap().join("empty-000000000000");
    std::fs::create_dir_all(&empty_group).unwrap();

    let c = store.clean().unwrap();
    assert_eq!(c.staging, vec![staging.clone()]);
    assert!(!staging.exists());
    assert_eq!(c.quarantine.len(), 1);
    assert!(!fit.folder().join(".attempts").exists());
    assert!(c.directories.contains(&empty_group));
    assert!(store.get(kept.id()).unwrap().is_some());
    assert!(store.get(fit.id()).unwrap().is_some());
}

#[test]
fn the_index_is_a_cache_rebuilt_from_the_records() {
    let root = TempRoot::new("index");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    let logs: Vec<_> = std::fs::read_dir(root.path().join(".index")).unwrap().flatten().collect();
    assert_eq!(logs.len(), 1);
    let text = std::fs::read_to_string(logs[0].path()).unwrap();
    assert!(text.contains(&s.id().key().to_hex()));
    drop(store);
    std::fs::remove_dir_all(root.path().join(".index")).unwrap();
    let store = open(root.path());
    assert!(store.get(s.id()).unwrap().is_some(), "a result is found without the index");
    assert_eq!(store.reindex().unwrap(), 1);
    assert!(root.path().join(".index").read_dir().unwrap().count() >= 1);
}

#[test]
fn the_commands_report_from_the_records() {
    let root = TempRoot::new("commands");
    let store = open(root.path());
    let i = labelled("base");
    let s = run_sim(&store, &i, 1);
    i.finish(&store).unwrap();
    let mut out = Vec::new();
    StoreCommand::List { kind: "sim".into() }.run(&store, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains(&s.id().key().short12()) && text.contains("completed") && text.contains("base"), "{text}");
    let mut out = Vec::new();
    StoreCommand::Show { handle: "@base".into() }.run(&store, &mut out).unwrap();
    assert!(String::from_utf8(out).unwrap().contains("asked for by invocation 1 (@base)"));
    let mut out = Vec::new();
    StoreCommand::Verify { handle: None }.run(&store, &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "every result verified\n");
    assert!(StoreCommand::List { kind: "nope".into() }.run(&store, &mut Vec::new()).is_err());
    for c in [StoreCommand::Status, StoreCommand::Clean, StoreCommand::Reindex] {
        c.run(&store, &mut Vec::new()).unwrap();
    }
}

/// `verify <handle>` checks the result an id prefix names, not another result
/// of the invocation that asked for it, and every result of an invocation a
/// label names.
#[test]
fn verify_checks_the_results_its_handle_names() {
    let root = TempRoot::new("verify-handle");
    let store = open(root.path());
    let i = labelled("both");
    let first = run_sim(&store, &i, 1);
    let second = run_sim(&store, &i, 2);
    i.finish(&store).unwrap();
    for s in [&first, &second] {
        let mut out = Vec::new();
        StoreCommand::Verify { handle: Some(s.id().key().to_hex()[..12].to_string()) }.run(&store, &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "every result verified\n");
    }
    std::fs::write(second.folder().join("traj.tsv"), b"t\tI\n0\t9\n").unwrap();
    let verify = |handle: String| {
        let mut out = Vec::new();
        StoreCommand::Verify { handle: Some(handle) }.run(&store, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    };
    let named = verify(second.id().key().to_hex()[..12].to_string());
    assert!(named.contains("damaged") && named.contains(&second.id().key().short12()), "{named}");
    let other = verify(first.id().key().to_hex()[..12].to_string());
    assert_eq!(other, "every result verified\n", "the first result is whole");
    let labelled = verify("@both".into());
    assert!(labelled.contains(&second.id().key().short12()), "a label checks every result of its invocation: {labelled}");
    let verdict = |handle: Option<String>| StoreCommand::Verify { handle }.run(&store, &mut Vec::new()).unwrap();
    assert_eq!(verdict(None), crate::Verdict::Damaged { results: 1 });
    assert_eq!(verdict(Some(first.id().key().to_hex())), crate::Verdict::Sound);
}

/// A path that names no result is refused, rather than reported verified
/// because nothing was checked.
#[test]
fn verify_refuses_a_path_that_names_no_result() {
    let root = TempRoot::new("verify-path");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    let group = s.folder().parent().unwrap().to_string_lossy().into_owned();
    for handle in [group, root.path().join("no-such-folder").to_string_lossy().into_owned()] {
        let r = StoreCommand::Verify { handle: Some(handle.clone()) }.run(&store, &mut Vec::new());
        assert!(matches!(r, Err(crate::Error::NoRecord { .. })), "{handle}: {r:?}");
    }
    let mut out = Vec::new();
    StoreCommand::Verify { handle: Some(s.folder().to_string_lossy().into_owned()) }.run(&store, &mut out).unwrap();
    assert_eq!(String::from_utf8(out).unwrap(), "every result verified\n");
    let i = labelled("gone");
    let gone = run_sim(&store, &i, 2);
    i.finish(&store).unwrap();
    std::fs::remove_dir_all(gone.folder()).unwrap();
    let r = StoreCommand::Verify { handle: Some("@gone".into()) }.run(&store, &mut Vec::new());
    assert!(matches!(r, Err(crate::Error::NotFound { .. })), "a label whose results are all gone: {r:?}");
}

#[test]
fn a_dead_lock_is_taken_over_by_exactly_one_of_many_takers() {
    let root = TempRoot::new("lock");
    let lock = root.path().join(".lock");
    std::fs::write(&lock, serde_json::to_vec(&crate::fsutil::LockFile { format: 1, owner: dead_owner() }).unwrap()).unwrap();
    let n = 8;
    let barrier = Arc::new(Barrier::new(n));
    let winners: Vec<bool> = (0..n)
        .map(|_| {
            let (lock, barrier) = (lock.clone(), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                let live = |o: &crate::Owner| crate::owner::local_liveness(o);
                match try_lock(&lock, &live).unwrap() {
                    LockAttempt::Acquired(g) => {
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        drop(g);
                        true
                    }
                    LockAttempt::Held(..) => false,
                }
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    assert_eq!(winners.iter().filter(|w| **w).count(), 1, "{winners:?}");
    assert!(!lock.exists());
}

#[test]
fn concurrent_claims_of_one_in_place_result_never_fail_with_an_io_error() {
    let root = TempRoot::new("concurrent");
    let store = Arc::new(open(root.path()));
    let barrier = Arc::new(Barrier::new(6));
    let outcomes: Vec<&'static str> = (0..6)
        .map(|_| {
            let (store, barrier) = (Arc::clone(&store), Arc::clone(&barrier));
            std::thread::spawn(move || {
                barrier.wait();
                let f = FitId { problem: "p".into(), method: "race".into() };
                match store.begin::<Fit>(&f, &(), &info(), Reuse::IfCompleted).unwrap() {
                    Begin::Claimed(run) => {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        run.finish().unwrap();
                        "claimed"
                    }
                    Begin::Busy(_) => "busy",
                    Begin::Reused(_) => "reused",
                }
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect();
    assert_eq!(outcomes.iter().filter(|o| **o == "claimed").count(), 1, "{outcomes:?}");
}

#[test]
fn a_timestamp_is_utc_to_the_millisecond() {
    use std::time::{Duration, UNIX_EPOCH};
    assert_eq!(crate::Timestamp::from_system(UNIX_EPOCH).0, "1970-01-01T00:00:00.000Z");
    assert_eq!(crate::Timestamp::from_system(UNIX_EPOCH + Duration::from_millis(1_763_630_043_120)).0, "2025-11-20T09:14:03.120Z");
    assert_eq!(crate::Timestamp::from_system(UNIX_EPOCH + Duration::from_secs(951_782_400)).0, "2000-02-29T00:00:00.000Z");
}

#[test]
fn a_scratch_store_is_removed_when_dropped() {
    let store = crate::Store::scratch(Build::Known(Engine { build: "s".into() }), kinds()).unwrap();
    let root = store.root().to_path_buf();
    run_sim(&store, &info(), 1);
    assert!(root.exists());
    drop(store);
    assert!(!root.exists());
}
