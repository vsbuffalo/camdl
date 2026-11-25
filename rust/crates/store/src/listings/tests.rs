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

/// Stores the model `text` and records it, in `info`, as the source of the
/// simulation of `seed`, which `info` lists.
fn run_with_source(store: &crate::Store, info: &crate::RunInfo, seed: u64, text: &str) -> (crate::Id<Sim>, crate::Id<Model>) {
    let s = run_sim(store, info, seed);
    let m = store
        .put::<Model>(crate::Hashed::new(ModelText(text.into())), &ModelAttributes { name: "m".into() }, &crate::Provenance::none(), info)
        .unwrap();
    info.record_source(s.id(), m.id()).unwrap();
    (*s.id(), *m.id())
}

#[test]
fn the_source_is_the_answering_invocations_with_no_fallback_and_others_named() {
    use crate::Recorded;
    let root = TempRoot::new("listings-source");
    let store = open(root.path());
    let a = labelled("a");
    let (sim, m1) = run_with_source(&store, &a, 8, "S -> I");
    a.finish(&store).unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(root.path().join(".invocations/1.json")).unwrap()).unwrap();
    assert_eq!(record["results"][0]["source"], serde_json::json!({ "kind": "compiled_model", "run_id": m1.key().to_hex() }));
    assert!(record["results"][1].get("source").is_none(), "a listing with no source has no field: {record:#}");
    let by_label = store.listings(&"@a".parse().unwrap()).unwrap();
    assert_eq!(by_label.source::<Sim, Model>(&sim), Recorded::One { source: m1, unfinished: false });

    run_sim(&store, &info(), 8);
    let by_id = id_handle(sim.key());
    assert_eq!(store.listings(&by_id).unwrap().source::<Sim, Model>(&sim), Recorded::None, "the newest recorded none, and no earlier one stands in");

    let c = info();
    let (_, m2) = run_with_source(&store, &c, 8, "S -> I -> R");
    c.finish(&store).unwrap();
    assert_eq!(store.listings(&by_id).unwrap().source::<Sim, Model>(&sim), Recorded::Several { latest: m2, others: vec![(1, m1)] });

    let other = info();
    run_sim(&store, &other, 9);
    let (unlisted, _) = super::super::tests::sim(10);
    let id = crate::Id::<Sim>::of(&store.inner.ambient_key, &unlisted);
    assert!(matches!(other.record_source(&id, &crate::Id::<Model>::from_key(sim.key())), Err(Error::NotListed { .. })));
}

/// An invocation records one source per result: the same source again is
/// accepted, another one is refused, and the first stays recorded.
#[test]
fn a_second_different_source_for_one_result_is_refused() {
    let root = TempRoot::new("listings-two-sources");
    let store = open(root.path());
    let i = info();
    let (sim, m1) = run_with_source(&store, &i, 11, "S -> I");
    i.record_source(&sim, &m1).unwrap();
    let m2 = store
        .put::<Model>(crate::Hashed::new(ModelText("S -> I -> R".into())), &ModelAttributes { name: "m".into() }, &crate::Provenance::none(), &i)
        .unwrap();
    let e = i.record_source(&sim, m2.id()).unwrap_err();
    assert!(matches!(e, Error::SourceDiffers { recorded, asked, .. } if recorded == m1.key() && asked == m2.id().key()), "{e}");
    i.finish(&store).unwrap();
    assert_eq!(store.listings(&id_handle(sim.key())).unwrap().source::<Sim, Model>(&sim), crate::Recorded::One { source: m1, unfinished: false });
}
