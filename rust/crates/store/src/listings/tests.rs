//! Which invocation answers for what a handle names.

use crate::tests::*;
use crate::{By, Error, Handle};

fn id_handle(key: crate::Key) -> Handle {
    Handle::IdPrefix(key.to_hex())
}

/// Rewrites `.invocations/<seq>.json` under `root` through `edit`.
fn edit_record(root: &std::path::Path, seq: u64, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = root.join(format!(".invocations/{seq}.json"));
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut v);
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
}

fn fail_first_result(v: &mut serde_json::Value) {
    v["results"][0]["outcome"] = "failed".into();
    v["results"][0]["failure"] = serde_json::json!({ "operation": null, "reason": "test" });
}

#[test]
fn a_result_one_invocation_listed_has_one_listing_and_it_answers() {
    let root = TempRoot::new("listings-one");
    let store = open(root.path());
    let s = run_sim(&store, &labelled("a"), 1);
    let l = store.listings(&id_handle(s.id().key())).unwrap();
    assert_eq!(l.by(), By::Result { run_id: s.id().key() });
    assert_eq!(l.listed().len(), 1);
    let answering = l.answering().unwrap();
    assert_eq!((answering.seq, answering.unfinished), (1, false));
    assert_eq!(answering.label.as_ref().map(crate::Label::as_str), Some("a"));
    assert_eq!(answering.attributes::<Sim>().unwrap().seed, 1);
    assert_eq!(store.listings(&s.folder().display().to_string().parse().unwrap()).unwrap().answering().unwrap().seq, 1);
}

#[test]
fn several_listings_are_newest_first_and_the_newest_answers() {
    let root = TempRoot::new("listings-several");
    let store = open(root.path());
    let s = run_sim(&store, &labelled("a"), 2);
    run_sim(&store, &info(), 2);
    run_sim(&store, &labelled("b"), 2);
    run_sim(&store, &info(), 2);
    let l = store.listings(&id_handle(s.id().key())).unwrap();
    let seqs: Vec<u64> = l.listed().iter().map(|l| l.seq).collect();
    assert_eq!(seqs, vec![4, 3, 2, 1]);
    assert_eq!(l.answering().unwrap().seq, 4);
    assert_eq!(l.labelled().unwrap().seq, 3, "the newest listing with a --label");
    assert_eq!(l.first().unwrap().seq, 1, "the oldest listing");
    assert_eq!(store.invocation(&id_handle(s.id().key())).unwrap().seq(), 4);
}

#[test]
fn a_label_names_one_invocation_which_answers_alone() {
    let root = TempRoot::new("listings-label");
    let store = open(root.path());
    run_sim(&store, &labelled("a"), 3);
    run_sim(&store, &labelled("b"), 3);
    run_sim(&store, &labelled("a"), 4);
    let l = store.listings(&"@a~1".parse().unwrap()).unwrap();
    assert_eq!(l.by(), By::Label { seq: 1 });
    assert_eq!(l.listed().len(), 1);
    assert_eq!(l.answering().unwrap().attributes::<Sim>().unwrap().seed, 3);
    assert!(matches!(store.listings(&"@none".parse().unwrap()), Err(Error::NotFound { .. })));
}

/// A listing whose outcome is not computed or reused is passed over, so the
/// newest listing that completed answers.
#[test]
fn a_failed_listing_is_passed_over() {
    let root = TempRoot::new("listings-failed");
    let s = run_sim(&open(root.path()), &labelled("ok"), 5);
    run_sim(&open(root.path()), &labelled("later"), 5);
    edit_record(root.path(), 2, fail_first_result);
    let store = open(root.path());
    let l = store.listings(&id_handle(s.id().key())).unwrap();
    let seqs: Vec<u64> = l.listed().iter().map(|l| l.seq).collect();
    assert_eq!(seqs, vec![1], "the failed listing in invocation 2 is not one");
    assert_eq!(store.invocation(&id_handle(s.id().key())).unwrap().seq(), 1);
}

/// A result no closed invocation listed computed or reused has no listing,
/// and no invocation answers for it.
#[test]
fn a_result_with_no_completed_listing_has_none() {
    let root = TempRoot::new("listings-none");
    let s = run_sim(&open(root.path()), &info(), 6);
    edit_record(root.path(), 1, fail_first_result);
    let store = open(root.path());
    let l = store.listings(&id_handle(s.id().key())).unwrap();
    assert!(l.listed().is_empty() && l.answering().is_none() && l.labelled().is_none() && l.first().is_none());
    assert!(matches!(store.invocation(&id_handle(s.id().key())), Err(Error::NotFound { .. })));
}

/// When no closed listing counts, the newest listing in a running record
/// answers, marked unfinished: a running record of another command is read
/// from the directory, since no snapshot holds it.
#[test]
fn a_running_listing_answers_marked_unfinished_when_no_closed_one_counts() {
    let root = TempRoot::new("listings-running");
    let s = run_sim(&open(root.path()), &info(), 7);
    run_sim(&open(root.path()), &labelled("slow"), 7);
    edit_record(root.path(), 1, fail_first_result);
    // Invocation 2 as it was while it ran: running, its result planned.
    edit_record(root.path(), 2, |v| {
        v["status"] = "running".into();
        v.as_object_mut().unwrap().remove("ended");
        v["results"][0]["outcome"] = "planned".into();
    });
    let store = open(root.path());
    let l = store.listings(&id_handle(s.id().key())).unwrap();
    let answering = l.answering().unwrap();
    assert_eq!((answering.seq, answering.unfinished), (2, true));
    assert_eq!(l.listed().len(), 1);
    assert_eq!(answering.attributes::<Sim>().unwrap().seed, 7);
}
