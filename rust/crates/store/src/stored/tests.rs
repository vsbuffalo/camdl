//! Kind-agnostic reads: a result located by its run id alone, and an ended
//! invocation's outputs.

use crate::tests::*;
use crate::{Begin, Error, Key, Reuse, Wanted};

/// A cell with its `obs` by-product, under `info`.
fn sim_with_obs(store: &Store, info: &RunInfo, seed: u64) -> crate::Stored<Sim> {
    let (id, labels) = sim(seed);
    let wanted = Wanted::<Sim>::none().with::<Obs>(ObsId { sim: id.record(), rate: Finite::new(0.3).unwrap() }, ());
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, info, Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(files(&["traj.tsv"], &["execution.json"]));
    run.result_file("traj.tsv").unwrap().write_all(format!("t\tI\n0\t{seed}\n").as_bytes()).unwrap();
    run.provenance_file("execution.json").unwrap().write_all(b"{}").unwrap();
    also_obs(&mut run).unwrap().result_file("obs.tsv").unwrap().write_all(b"o").unwrap();
    run.finish().unwrap()
}

#[test]
fn a_result_is_located_by_its_run_id_with_its_kind_and_files() {
    let root = TempRoot::new("locate");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 1);
    let located = store.locate(s.id().key()).unwrap();
    assert_eq!(located.kind(), "sim");
    assert_eq!(located.folder(), s.folder());
    assert_eq!(located.upstream(), None);
    let digest = Key::digest_bytes(b"t\tI\n0\t1\n");
    assert_eq!(located.result_files().into_iter().collect::<Vec<_>>(), vec![("traj.tsv", (digest, 8))]);
    assert_eq!(located.file("traj.tsv").unwrap(), (s.folder().join("traj.tsv"), digest));
    assert_eq!(located.read("traj.tsv").unwrap(), b"t\tI\n0\t1\n");
    assert!(matches!(located.file("absent.tsv"), Err(Error::BadFileName { .. })));
    // The same answer from the handle forms that name one result.
    let by_path: crate::Handle = s.folder().display().to_string().parse().unwrap();
    assert_eq!(store.locate_handle(&by_path).unwrap().run_id(), s.id().key());
    let by_prefix = crate::Handle::IdPrefix(s.id().key().to_hex()[..12].to_string());
    assert_eq!(store.locate_handle(&by_prefix).unwrap().run_id(), s.id().key());
}

#[test]
fn a_by_product_is_located_with_its_upstream_and_a_nested_result_without_one() {
    let root = TempRoot::new("locate-bp");
    let store = open(root.path());
    let i = info();
    let s = sim_with_obs(&store, &i, 2);
    let inv = i.finish(&store).unwrap();
    let obs = inv.file.results.iter().find(|r| r.kind == "obs").unwrap().run_id;
    let located = store.locate(obs).unwrap();
    assert_eq!(located.kind(), "obs");
    assert_eq!(located.upstream(), Some(s.id().key()));

    let fit_id = FitId { problem: "p".into(), method: "mh".into() };
    let Begin::Claimed(mut fit) = store.begin::<Fit>(&fit_id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    fit.expect(expecting(&["draws.tsv"]));
    fit.result_file("draws.tsv").unwrap().write_all(b"d").unwrap();
    let fit = fit.finish().unwrap();
    let pred = PredId { fit: fit.dep("draws.tsv").unwrap(), n: 1 };
    let Begin::Claimed(mut run) = store.begin::<Pred>(&pred, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["p.tsv"]));
    run.result_file("p.tsv").unwrap().write_all(b"p").unwrap();
    let pred = run.finish().unwrap();
    assert_eq!(store.locate(pred.id().key()).unwrap().upstream(), None);
}

#[test]
fn an_absent_running_or_damaged_result_is_not_located() {
    let root = TempRoot::new("locate-not");
    let store = open(root.path());
    assert!(matches!(store.locate(Key::digest_bytes(b"nothing")), Err(Error::NotFound { .. })));

    let fit_id = FitId { problem: "p".into(), method: "if2".into() };
    let Begin::Claimed(fit) = store.begin::<Fit>(&fit_id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    match store.locate(fit.id().key()) {
        Err(Error::NotCompleted { status, .. }) => assert_eq!(status, "running"),
        other => panic!("a running result is not completed: {other:?}"),
    }
    drop(fit);

    let s = run_sim(&store, &info(), 3);
    let located = store.locate(s.id().key()).unwrap();
    std::fs::write(s.folder().join("traj.tsv"), b"t\tI\n0\t9\n").unwrap();
    assert!(matches!(located.read("traj.tsv"), Err(Error::Damaged { .. })));
}

#[test]
fn a_label_locates_its_invocations_one_result_and_refuses_several() {
    let root = TempRoot::new("locate-label");
    let store = open(root.path());
    let one = labelled("one");
    let s = run_sim(&store, &one, 4);
    one.finish(&store).unwrap();
    assert_eq!(store.locate_handle(&"@one".parse().unwrap()).unwrap().run_id(), s.id().key());
    let two = labelled("two");
    run_sim(&store, &two, 5);
    run_sim(&store, &two, 6);
    two.finish(&store).unwrap();
    assert!(matches!(store.locate_handle(&"@two".parse().unwrap()), Err(Error::NotOneResult { .. })));
}

#[test]
fn outputs_list_each_distinct_result_in_order_with_by_products_and_result_files_only() {
    let root = TempRoot::new("outputs");
    let store = open(root.path());
    let i = info();
    let a = sim_with_obs(&store, &i, 7);
    let b = run_sim(&store, &i, 8);
    // Asked for again in the same invocation: listed twice, output once.
    run_sim(&store, &i, 8);
    let inv = i.finish(&store).unwrap();
    let outputs = inv.outputs(&store).unwrap();
    let got: Vec<(&str, Key, Option<Key>)> = outputs.iter().map(|o| (o.kind.as_str(), o.run_id, o.upstream)).collect();
    let obs = inv.file.results.iter().find(|r| r.kind == "obs").unwrap().run_id;
    assert_eq!(got, vec![("sim", a.id().key(), None), ("obs", obs, Some(a.id().key())), ("sim", b.id().key(), None)]);
    // The provenance file `execution.json` is not a result file.
    assert_eq!(outputs[0].files.keys().collect::<Vec<_>>(), vec!["traj.tsv"]);
    assert_eq!(outputs[0].files["traj.tsv"], crate::OutputFile { sha256: Key::digest_bytes(b"t\tI\n0\t7\n"), bytes: 8 });
    assert_eq!(outputs[0].labels, serde_json::json!({ "model": "sir", "seed": 7 }));
}

#[test]
fn outputs_refuse_an_invocation_with_a_result_that_did_not_complete() {
    let root = TempRoot::new("outputs-failed");
    let store = open(root.path());
    let i = info();
    let ok = run_sim(&store, &i, 9);
    let (id, labels) = sim(10);
    let _ = store.run::<Sim, String>(&id, &labels, &i, Reuse::IfCompleted, expecting(&[]), "simulating", |_| Err("no".into()));
    let inv = i.finish(&store).unwrap();
    match inv.outputs(&store) {
        Err(Error::MemberNotCompleted { members, .. }) => {
            assert_eq!(members.len(), 1);
            assert_ne!(members[0], ok.id().key());
        }
        other => panic!("a failed member is refused: {other:?}"),
    }
}
