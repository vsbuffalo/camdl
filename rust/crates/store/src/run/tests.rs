//! By-products: committed with their upstream, produced later in reproducing
//! mode, and failed with it.

use crate::tests::*;
use crate::{Begin, Error, Reuse, Wanted};

fn obs_of(id: &SimId, rate: f64) -> ObsId {
    ObsId { sim: id.record(), rate: Finite::new(rate).unwrap() }
}

fn write_sim(run: &mut crate::Run<Sim>, seed: u64) {
    run.expect(expecting(&["traj.tsv"]));
    run.result_file("traj.tsv").unwrap().write_all(format!("t\tI\n0\t{seed}\n").as_bytes()).unwrap();
}

/// Writes the wanted observations of `run`, declaring them.
fn write_obs(run: &mut crate::Run<Sim>) {
    let mut obs = run.also::<Obs>().unwrap().unwrap();
    obs.expect(expecting(&["obs.tsv"]));
    obs.result_file("obs.tsv").unwrap().write_all(b"o").unwrap();
}

#[test]
fn a_by_product_wanted_with_its_upstream_is_committed_with_it() {
    let root = TempRoot::new("bp-together");
    let store = open(root.path());
    let (id, labels) = sim(1);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.3), ());
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    assert!(!run.reproducing());
    write_sim(&mut run, 1);
    let mut obs = run.also::<Obs>().unwrap().expect("wanted");
    obs.expect(expecting(&["obs.tsv"]));
    obs.result_file("obs.tsv").unwrap().write_all(b"t\tcases\n").unwrap();
    assert!(run.also::<Obs>().unwrap().is_none(), "a by-product's part is given once");
    let s = run.finish().unwrap();
    let obs_dirs: Vec<_> = std::fs::read_dir(s.folder().join("obs")).unwrap().flatten().map(|e| e.path()).collect();
    assert_eq!(obs_dirs.len(), 1);
    let ident: serde_json::Value = serde_json::from_slice(&std::fs::read(obs_dirs[0].join("identity.json")).unwrap()).unwrap();
    assert_eq!(ident["parent"]["relation"], "upstream");
    assert_eq!(ident["parent"]["run_id"], s.id().key().to_hex());
    assert!(store.get(s.id()).unwrap().is_some(), "its by-products do not damage the upstream");
    let obs_id = crate::Id::<Obs>::of(&store.inner.ambient_key, &obs_of(&id, 0.3));
    assert!(store.get(&obs_id).unwrap().is_some());
    assert!(matches!(store.begin_with(&id, &labels, &wanted, &info(), Reuse::IfCompleted).unwrap(), Begin::Reused(_)));
}

#[test]
fn a_by_product_asked_for_later_reproduces_the_upstream_without_touching_it() {
    let root = TempRoot::new("bp-later");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 2);
    let before = std::fs::read(s.folder().join("run.json")).unwrap();
    let (id, labels) = sim(2);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.5), ());
    let info = info();
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info, Reuse::IfCompleted).unwrap() else { panic!() };
    assert!(run.reproducing());
    write_sim(&mut run, 2);
    run.progress(&"reproducing").unwrap();
    write_obs(&mut run);
    let back = run.finish().unwrap();
    assert_eq!(back.folder(), s.folder());
    assert_eq!(std::fs::read(s.folder().join("run.json")).unwrap(), before, "the stored upstream is not written");
    assert!(store.get(s.id()).unwrap().is_some());
    let staging: Vec<_> = std::fs::read_dir(s.folder().join(".staging")).map(|d| d.flatten().count()).into_iter().collect();
    assert!(staging.iter().all(|n| *n == 0), "nothing is left in staging");
    let inv = info.finish(&store).unwrap();
    let outcomes: Vec<_> = inv.record.listings().iter().map(|r| (r.kind, r.outcome.cloned())).collect();
    assert_eq!(outcomes, vec![("sim", Some(crate::record::Outcome::Reused)), ("obs", Some(crate::record::Outcome::Computed))]);
}

#[test]
fn a_reproduced_upstream_that_differs_is_divergent_and_its_files_are_kept() {
    let root = TempRoot::new("bp-divergent");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 3);
    let (id, labels) = sim(3);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.5), ());
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["traj.tsv"]));
    run.result_file("traj.tsv").unwrap().write_all(b"something else").unwrap();
    write_obs(&mut run);
    match run.finish() {
        Err(Error::Divergent { file, .. }) => assert_eq!(file, "traj.tsv"),
        other => panic!("{other:?}"),
    }
    assert!(!s.folder().join("obs").exists(), "the by-product is failed");
    let q: Vec<String> = std::fs::read_dir(root.path().join(".quarantine")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(q, vec![".reproduce".to_string()]);
    assert!(store.get(s.id()).unwrap().is_some(), "the stored upstream is untouched");
}

#[test]
fn a_reproduced_upstream_that_skips_a_stored_file_is_divergent() {
    let root = TempRoot::new("bp-skip");
    let store = open(root.path());
    run_sim(&store, &info(), 4);
    let (id, labels) = sim(4);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.5), ());
    let Begin::Claimed(run) = store.begin_with(&id, &labels, &wanted, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    assert!(matches!(run.finish(), Err(Error::Divergent { .. })));
}

#[test]
fn a_failed_upstream_fails_its_by_products() {
    let root = TempRoot::new("bp-fail");
    let store = open(root.path());
    let (id, labels) = sim(5);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.5), ());
    let info = info();
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info, Reuse::IfCompleted).unwrap() else { panic!() };
    write_obs(&mut run);
    run.fail("simulating", "a negative rate").unwrap();
    assert!(result_folders(root.path()).is_empty());
    let inv = info.finish(&store).unwrap();
    assert!(inv.record.listings().iter().all(|r| matches!(r.outcome, Some(crate::record::Outcome::Failed(_)))));
}

#[test]
fn an_in_place_upstream_commits_its_by_products_after_itself() {
    let root = TempRoot::new("bp-inplace");
    let store = open(root.path());
    let f = FitId { problem: "p".into(), method: "if2".into() };
    let wanted = Wanted::<Fit>::none().with::<Trace>(TraceId { fit: f.record() }, ());
    let Begin::Claimed(mut run) = store.begin_with(&f, &(), &wanted, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["draws.tsv"]));
    run.result_file("draws.tsv").unwrap().write_all(b"d").unwrap();
    let mut t = run.also::<Trace>().unwrap().unwrap();
    t.expect(expecting(&["trace.tsv"]));
    t.result_file("trace.tsv").unwrap().write_all(b"t").unwrap();
    let staged = result_folders(root.path());
    assert_eq!(staged.len(), 1, "a by-product is not visible before its upstream commits: {staged:?}");
    let s = run.finish().unwrap();
    assert!(s.folder().join("trace").is_dir());
    let trace_id = crate::Id::<Trace>::of(&store.inner.ambient_key, &TraceId { fit: f.record() });
    assert!(store.get(&trace_id).unwrap().is_some());
    assert!(store.get(s.id()).unwrap().is_some());
}

#[test]
fn a_divergence_from_an_extra_file_leaves_the_stored_upstream_reused() {
    let root = TempRoot::new("bp-extra");
    let store = open(root.path());
    run_sim(&store, &info(), 6);
    let (id, labels) = sim(6);
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.5), ());
    let info = info();
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info, Reuse::IfCompleted).unwrap() else { panic!() };
    write_sim(&mut run, 6);
    run.result_file("extra.tsv").unwrap().write_all(b"x").unwrap();
    assert!(matches!(run.finish(), Err(Error::Divergent { .. })));
    let inv = info.finish(&store).unwrap();
    assert_eq!(inv.record.listings()[0].outcome, Some(&crate::record::Outcome::Reused));
}

/// A by-product's record lists the reads of the run that wrote it, whether
/// it was produced with its upstream or later, in reproducing mode.
#[test]
fn a_by_product_records_the_reads_made_to_write_it() {
    let root = TempRoot::new("bp-reads");
    let store = open(root.path());
    let model = store
        .put::<Model>(Hashed::new(ModelText("m".into())), &ModelAttributes { name: "m".into() }, &crate::Provenance::none(), &info())
        .unwrap()
        .dep(Model::FILE)
        .unwrap();
    let reads_of_obs = |s: &crate::Stored<Sim>| -> serde_json::Value {
        let dir = std::fs::read_dir(s.folder().join("obs")).unwrap().flatten().next().unwrap().path();
        let run: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
        run["reads"].clone()
    };
    for (seed, later) in [(5, false), (6, true)] {
        if later {
            // The stored upstream read the model too, so a by-product added
            // later is written from a run whose own reads repeat it.
            let (id, labels) = sim(seed);
            let Begin::Claimed(mut run) = store.begin(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
            run.load(&model).unwrap();
            write_sim(&mut run, seed);
            run.finish().unwrap();
        }
        let (id, labels) = sim(seed);
        let wanted = Wanted::<Sim>::none().with::<Obs>(obs_of(&id, 0.3), ());
        let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
        assert_eq!(run.reproducing(), later);
        run.load(&model).unwrap();
        write_sim(&mut run, seed);
        write_obs(&mut run);
        let s = run.finish().unwrap();
        let reads = reads_of_obs(&s);
        assert_eq!(reads.as_array().map(Vec::len), Some(1), "later = {later}: each read is recorded once: {reads}");
        assert_eq!(reads[0]["kind"], "compiled_model");
    }
}
