//! Liveness read from another host: by the beat counter alone, never by a
//! process check or a timestamp.

use super::*;
use crate::tests::{open, TempRoot};

fn elsewhere() -> Owner {
    Owner { host: "another-host".into(), pid: 4242, started: Timestamp("2026-11-20T09:14:03.120Z".into()) }
}

fn write_beat(root: &Path, owner: &Owner, beat: u64) {
    let b = BeatFile {
        format: BEAT_FORMAT,
        host: owner.host.clone(),
        pid: owner.pid,
        started: owner.started.clone(),
        beat,
        invocation: None,
    };
    crate::fsutil::write_json_atomic(&beat_path(root, owner), &b).unwrap();
}

const WAIT: Duration = Duration::from_millis(300);

#[test]
fn an_owner_on_another_host_with_no_beat_file_is_unknown() {
    let root = TempRoot::new("live-unknown");
    let store = open(root.path());
    assert!(!elsewhere().is_this_host());
    assert_eq!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Unknown);
}

#[test]
fn an_unchanged_beat_is_stale_after_the_wait() {
    let root = TempRoot::new("live-stale");
    let store = open(root.path());
    std::fs::create_dir_all(root.path().join(".owners")).unwrap();
    write_beat(root.path(), &elsewhere(), 7);
    let started = Instant::now();
    assert!(matches!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Stale { .. }));
    assert!(started.elapsed() >= WAIT, "the reader waited on its own clock before calling it stale");
}

#[test]
fn a_beat_that_changes_during_the_wait_is_alive() {
    let root = TempRoot::new("live-alive");
    let store = open(root.path());
    std::fs::create_dir_all(root.path().join(".owners")).unwrap();
    write_beat(root.path(), &elsewhere(), 7);
    let r = root.path().to_path_buf();
    let bump = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        write_beat(&r, &elsewhere(), 8);
    });
    assert_eq!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Alive);
    bump.join().unwrap();
}

/// A long-lived reader keeps the last counter it saw: a counter that moved
/// since is alive at once, and one unchanged for the whole wait is stale at
/// once.
#[test]
fn a_reader_remembers_the_counter_it_saw() {
    let root = TempRoot::new("live-memory");
    let store = open(root.path());
    std::fs::create_dir_all(root.path().join(".owners")).unwrap();
    write_beat(root.path(), &elsewhere(), 1);
    assert!(matches!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Stale { .. }));
    write_beat(root.path(), &elsewhere(), 2);
    let t = Instant::now();
    assert_eq!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Alive);
    assert!(t.elapsed() < WAIT, "a counter that moved since the last look needs no wait");
    std::thread::sleep(WAIT);
    let t = Instant::now();
    assert!(matches!(store.liveness(&elsewhere(), WAIT).unwrap(), Liveness::Stale { .. }));
    assert!(t.elapsed() < WAIT, "a counter unchanged for the wait is stale without waiting again");
}
