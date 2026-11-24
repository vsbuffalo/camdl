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
    assert_eq!(store.locate(pred.id().key()).unwrap().parent(), Some(fit.id().key()), "a nested result names its parent");
    assert_eq!(located.parent(), Some(s.id().key()), "a by-product's parent is its upstream");
    assert_eq!(store.locate(fit.id().key()).unwrap().parent(), None);
}

/// A completed fit run of `method`, writing `draws.tsv`.
fn finished_fit(store: &Store, method: &str) -> crate::Stored<Fit> {
    let fit_id = FitId { problem: "p".into(), method: method.into() };
    let Begin::Claimed(mut fit) = store.begin::<Fit>(&fit_id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    fit.expect(expecting(&["draws.tsv"]));
    fit.result_file("draws.tsv").unwrap().write_all(b"d").unwrap();
    fit.finish().unwrap()
}

/// Begins the prediction `n` inside `fit`, writing `p.tsv`.
fn begin_pred(store: &Store, fit: &crate::Stored<Fit>, n: u64) -> crate::Run<Pred> {
    let id = PredId { fit: fit.dep("draws.tsv").unwrap(), n };
    let Begin::Claimed(mut run) = store.begin::<Pred>(&id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["p.tsv"]));
    run.result_file("p.tsv").unwrap().write_all(format!("p{n}").as_bytes()).unwrap();
    run
}

#[test]
fn the_children_of_a_result_are_its_completed_results_of_one_kind_in_folder_order() {
    let root = TempRoot::new("children");
    let store = open(root.path());
    let fit = finished_fit(&store, "mh");
    let other = finished_fit(&store, "pmmh");
    // Finished out of folder order: n4 before n1.
    let n4 = begin_pred(&store, &fit, 4).finish().unwrap();
    let n1 = begin_pred(&store, &fit, 1).finish().unwrap();
    let damaged = begin_pred(&store, &fit, 2).finish().unwrap();
    std::fs::write(damaged.folder().join("p.tsv"), b"edited").unwrap();
    drop(begin_pred(&store, &fit, 3));
    let running = begin_pred(&store, &fit, 5);
    begin_pred(&store, &other, 6).finish().unwrap();
    // A folder whose identity.json does not read is skipped, not an error.
    let junk = fit.folder().join("predictions").join("junk");
    std::fs::create_dir_all(&junk).unwrap();
    std::fs::write(junk.join("identity.json"), b"{").unwrap();

    let children = store.children_of::<Pred, Fit>(&fit).unwrap();
    let keys: Vec<Key> = children.iter().map(|c| c.id().key()).collect();
    assert_eq!(keys, vec![n1.id().key(), n4.id().key()], "completed only, damaged and failed excluded, in folder order");
    assert_eq!(store.children_of::<Pred, Fit>(&other).unwrap().len(), 1, "another parent's children are its own");
    drop(running);

    // A by-product is a child of its upstream, of its own kind only.
    let s = sim_with_obs(&store, &info(), 7);
    assert_eq!(store.children_of::<Obs, Sim>(&s).unwrap().len(), 1);
    assert!(store.children_of::<Pred, Sim>(&s).unwrap().is_empty());
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

#[test]
fn a_located_result_lists_its_provenance_files_and_opens_as_its_kind() {
    let root = TempRoot::new("locate-typed");
    let store = open(root.path());
    let s = sim_with_obs(&store, &info(), 11);
    let located = store.locate(s.id().key()).unwrap();
    assert_eq!(located.provenance_files().unwrap().into_iter().collect::<Vec<_>>(), vec![("execution.json", Key::digest_bytes(b"{}"))]);
    let typed = store.stored_as::<Sim>(&located).unwrap();
    assert_eq!(typed.id().key(), s.id().key());
    assert_eq!(typed.folder(), s.folder());
    assert!(matches!(store.stored_as::<Fit>(&located), Err(Error::WrongKind { .. })), "a sim does not open as a fit");
}

#[test]
fn a_damaged_by_product_still_names_its_upstream() {
    let root = TempRoot::new("upstream-of");
    let store = open(root.path());
    let i = info();
    let s = sim_with_obs(&store, &i, 12);
    let inv = i.finish(&store).unwrap();
    let obs = inv.file.results.iter().find(|r| r.kind == "obs").unwrap().run_id;
    let folder = store.locate(obs).unwrap().folder().to_path_buf();
    std::fs::write(folder.join("obs.tsv"), b"other").unwrap();
    assert!(store.locate(obs).is_err() || store.locate(obs).unwrap().read("obs.tsv").is_err(), "the by-product is damaged");
    std::fs::remove_file(folder.join("run.json")).unwrap();
    assert!(store.locate(obs).is_err(), "a by-product without its record does not locate");
    assert_eq!(store.upstream_of(obs).unwrap(), Some(s.id().key()));
    assert_eq!(store.upstream_of(s.id().key()).unwrap(), None, "a cell is no by-product");
    assert!(matches!(store.upstream_of(Key::digest_bytes(b"none")), Err(Error::NotFound { .. })));
}
