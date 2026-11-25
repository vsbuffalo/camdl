//! Importing a copy of another store: what is placed, where, what is
//! recorded, and what is refused with nothing placed.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use crate::tests::*;
use crate::{Begin, Error, Handle, Key, Reuse, RunRef, Wanted};

/// What another writer does at a destination just before `place` renames a
/// result there.
type Interloper = Box<dyn Fn(&Path)>;

thread_local! {
    /// How many more results `place` moves before it stops as a killed
    /// process would; `None` for no stop.
    static MOVES_LEFT: Cell<Option<usize>> = const { Cell::new(None) };
    static INTERLOPER: RefCell<Option<Interloper>> = const { RefCell::new(None) };
}

/// Called by `place` before each rename: once the moves allowed by
/// [`MOVES_LEFT`] are used it panics, so no code of the import runs after
/// it, as when the process is killed; and it runs the [`INTERLOPER`] on the
/// destination.
pub(crate) fn before_move(dest: &Path) -> crate::Result<()> {
    MOVES_LEFT.with(|m| match m.get() {
        Some(0) => panic!("the import stops here, as a killed process would"),
        Some(n) => m.set(Some(n - 1)),
        None => {}
    });
    INTERLOPER.with(|i| {
        if let Some(f) = i.borrow().as_ref() {
            f(dest)
        }
    });
    Ok(())
}

/// The results the other store holds, by run id.
struct Remote {
    sim: Key,
    obs: Key,
    fit: Key,
    pred: Key,
}

/// Fills `store` with a simulation and its observations, a fit, and a
/// prediction inside the fit: the simulation and the fit under one
/// invocation labelled `remote`, the prediction under a second.
fn fill(store: &Store) -> Remote {
    let i = labelled("remote");
    let (id, attributes) = sim(1);
    let wanted = Wanted::<Sim>::none().with::<Obs>(ObsId { sim: id.record(), rate: Finite::new(0.3).unwrap() }, ());
    let Begin::Claimed(mut run) = store.begin_with(&id, &attributes, &wanted, &i, Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(files(&["traj.tsv"], &["execution.json"]));
    run.result_file("traj.tsv").unwrap().write_all(b"t\tI\n0\t1\n").unwrap();
    run.provenance_file("execution.json").unwrap().write_all(b"{\"host\":\"remote\"}").unwrap();
    also_obs(&mut run).unwrap().result_file("obs.tsv").unwrap().write_all(b"o").unwrap();
    let s = run.finish().unwrap();
    let fit = fit_in(store, &i, b"d");
    let inv = i.finish(store).unwrap();
    let obs = inv.record.listings().iter().find(|r| r.kind == "obs").unwrap().run_id;
    let p = labelled("forecast");
    let pred = pred_in(store, &p, &fit);
    p.finish(store).unwrap();
    Remote { sim: s.id().key(), obs, fit: fit.id().key(), pred: pred.id().key() }
}

fn fit_in(store: &Store, info: &RunInfo, draws: &[u8]) -> crate::Stored<Fit> {
    let fit_id = FitId { problem: "p".into(), method: "mh".into() };
    let Begin::Claimed(mut run) = store.begin::<Fit>(&fit_id, &(), info, Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["draws.tsv"]));
    run.result_file("draws.tsv").unwrap().write_all(draws).unwrap();
    run.finish().unwrap()
}

fn pred_in(store: &Store, info: &RunInfo, fit: &crate::Stored<Fit>) -> crate::Stored<Pred> {
    let id = PredId { fit: fit.dep("draws.tsv").unwrap(), n: 1 };
    let Begin::Claimed(mut run) = store.begin::<Pred>(&id, &(), info, Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(expecting(&["p.tsv"]));
    run.result_file("p.tsv").unwrap().write_all(b"p").unwrap();
    run.finish().unwrap()
}

/// What `pull` hands over: the other store's `store.json`, its invocation
/// records and its result trees, without its index, beats or staging.
fn pull(remote: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(remote).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') && name != ".invocations" {
            continue;
        }
        if e.path().is_dir() {
            crate::fsutil::copy_tree(&e.path(), &to.join(&name)).unwrap();
        } else {
            std::fs::copy(e.path(), to.join(&name)).unwrap();
        }
    }
}

/// Every file below `root`, by its path relative to `root`, with its bytes.
fn tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn go(dir: &Path, root: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                go(&p, root, out);
            } else {
                out.insert(crate::fsutil::rel_key(root, &p), std::fs::read(&p).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    go(root, root, &mut out);
    out
}

/// A local store under `root` (build `b1`) and a copy of a filled store of
/// build `remote`, at `<root>/../pulled`.
struct Scene {
    remote_root: TempRoot,
    local_root: TempRoot,
    pull_root: TempRoot,
    remote: Remote,
}

impl Scene {
    fn new(tag: &str) -> Scene {
        let remote_root = TempRoot::new(&format!("{tag}-remote"));
        let remote = fill(&open_build(remote_root.path(), "remote"));
        let pull_root = TempRoot::new(&format!("{tag}-pull"));
        pull(remote_root.path(), &pull_root.path().join("copy"));
        Scene { remote_root, local_root: TempRoot::new(&format!("{tag}-local")), pull_root, remote }
    }

    fn copy(&self) -> PathBuf {
        self.pull_root.path().join("copy")
    }

    /// The copy's folder of the result `key`.
    fn copied(&self, key: Key) -> PathBuf {
        result_folders(&self.copy())
            .into_iter()
            .map(|r| self.copy().join(r))
            .find(|d| crate::storage::identity_of(d).is_some_and(|f| f.run_id == key))
            .unwrap()
    }
}

#[test]
fn import_places_and_records_a_pulled_store_with_its_nested_results() {
    let s = Scene::new("import");
    // The other store's process state in the fit's folder: a lock and a
    // by-product claim, each naming an owner on another host.
    let fit_copy = s.copied(s.remote.fit);
    std::fs::write(fit_copy.join(".lock"), b"{}").unwrap();
    std::fs::create_dir_all(fit_copy.join(".staging").join("claim")).unwrap();
    // And files no digest covers: an earlier attempt, a record's temporary,
    // and a folder holding only an in-place claim's lock.
    std::fs::create_dir_all(fit_copy.join(".attempts").join("1")).unwrap();
    std::fs::write(fit_copy.join(".attempts").join("1").join("attempt.json"), b"{}").unwrap();
    std::fs::write(fit_copy.join(".identity.json.x.tmp"), b"{}").unwrap();
    let claim = s.copied(s.remote.pred).parent().unwrap().join("n2-000000000000");
    std::fs::create_dir_all(&claim).unwrap();
    std::fs::write(claim.join(".lock"), b"{}").unwrap();
    let local = open(s.local_root.path());
    let i = labelled("pulled");
    let got = local.import(&s.copy(), &i).unwrap();
    let inv = i.finish(&local).unwrap();

    let r = &s.remote;
    let mut placed = got.placed.clone();
    placed.sort();
    let mut all = vec![r.sim, r.obs, r.fit, r.pred];
    all.sort();
    assert_eq!(placed, all);
    // The index gains one line per placed result. Read before any lookup,
    // which would rebuild it from the records.
    let lines: Vec<serde_json::Value> = tree(&s.local_root.path().join(".index"))
        .values()
        .flat_map(|b| String::from_utf8(b.clone()).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect::<Vec<_>>())
        .filter(|l: &serde_json::Value| l["event"] == "completed")
        .collect();
    for key in &all {
        assert_eq!(lines.iter().filter(|l| l["run_id"] == key.to_hex()).count(), 1, "{key}");
    }
    assert!(got.present.is_empty());
    // Each result is at the path it had in the other store, which is under
    // the other build's directory.
    assert_eq!(result_folders(s.local_root.path()), result_folders(s.remote_root.path()));
    let sim_dir = local.locate(r.sim).unwrap().folder().to_path_buf();
    let top = sim_dir.strip_prefix(s.local_root.path()).unwrap().components().next().unwrap();
    let top = top.as_os_str().to_string_lossy().into_owned();
    let remote_top = result_folders(s.remote_root.path())[0].split('/').next().unwrap().to_string();
    assert_eq!(top, remote_top);
    assert!(!top.ends_with(&local_build_suffix()), "nothing lands in this build's directory");
    // Nested results sit inside their parents.
    let fit_dir = local.locate(r.fit).unwrap().folder().to_path_buf();
    assert!(local.locate(r.pred).unwrap().folder().starts_with(&fit_dir));
    assert!(!fit_dir.join(".lock").exists() && !fit_dir.join(".staging").exists(), "the other store's process state is dropped");
    assert!(!fit_dir.join(".attempts").exists() && !fit_dir.join(".identity.json.x.tmp").exists(), "unverified files are dropped");
    let pred_dir = local.locate(r.pred).unwrap().folder().to_path_buf();
    assert!(!pred_dir.parent().unwrap().join("n2-000000000000").exists(), "a lock-only folder is dropped");
    assert!(local.locate(r.obs).unwrap().folder().starts_with(&sim_dir));
    assert_eq!(local.locate(r.obs).unwrap().upstream(), Some(r.sim));
    assert!(local.verify(None).unwrap().damaged.is_empty());

    // One invocation lists them, parents first, reused, with the attributes the
    // other store's records gave them.
    assert_eq!(inv.label(), Some("pulled"));
    let kinds: Vec<&str> = inv.record.listings().iter().map(|e| e.kind).collect();
    let pos = |k: &str| kinds.iter().position(|x| *x == k).unwrap();
    assert!(pos("fit") < pos("prediction") && pos("sim") < pos("obs"));
    assert!(inv.record.listings().iter().all(|e| e.outcome == Some(&crate::record::Outcome::Reused)));
    assert_eq!(inv.members::<Sim>().unwrap()[0].1, SimAttributes { model: "sir".into(), seed: 1 });
    // `@pulled` names them.
    let pred: RunRef<Pred> = "@pulled".parse().unwrap();
    assert_eq!(local.resolve(&pred).unwrap().id().key(), r.pred);
    let h: Handle = "@pulled".parse().unwrap();
    assert_eq!(local.invocation(&h).unwrap().seq(), inv.seq());
    // The copy is consumed and the staging left empty.
    assert!(!s.copy().exists());
    let staging = s.local_root.path().join(".staging");
    assert_eq!(std::fs::read_dir(&staging).unwrap().count(), 0);
}

/// The 12-hex suffix of the directory of the local test build, `b1`.
fn local_build_suffix() -> String {
    let root = TempRoot::new("suffix");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 9);
    let top = s.folder().strip_prefix(root.path()).unwrap().components().next().unwrap().as_os_str().to_string_lossy().into_owned();
    top.rsplit('-').next().unwrap().to_string()
}

#[test]
fn import_writes_nothing_outside_staging_the_placed_folders_and_the_records() {
    let s = Scene::new("import-writes");
    let local = open(s.local_root.path());
    let i = labelled("pulled");
    local.import(&s.copy(), &i).unwrap();
    i.finish(&local).unwrap();
    let placed: Vec<String> = result_folders(s.local_root.path());
    for path in tree(s.local_root.path()).keys() {
        let ok = path == "store.json"
            || [".index/", ".invocations/", ".owners/"].iter().any(|p| path.starts_with(p))
            || placed.iter().any(|f| path.starts_with(&format!("{f}/")));
        assert!(ok, "{path} was written outside the placed folders and the store's records");
    }
}

#[test]
fn import_refuses_a_damaged_copy_and_places_nothing() {
    let s = Scene::new("import-damaged");
    let local = open(s.local_root.path());
    // One byte of one file changed, its size and modification time kept, so
    // only its digest tells.
    let damaged = s.copied(s.remote.pred).join("p.tsv");
    let mtime = std::fs::metadata(&damaged).unwrap().modified().unwrap();
    std::fs::write(&damaged, b"q").unwrap();
    std::fs::File::options().write(true).open(&damaged).unwrap().set_modified(mtime).unwrap();
    let before_copy = tree(&s.copy());
    let before_local = tree(s.local_root.path());
    let i = labelled("pulled");
    let e = local.import(&s.copy(), &i).unwrap_err();
    match &e {
        Error::ImportRefused { folder, reason } => {
            assert_eq!(folder, &s.copied(s.remote.pred));
            assert!(reason.contains("p.tsv"), "{reason}");
        }
        other => panic!("{other}"),
    }
    drop(i);
    assert!(result_folders(s.local_root.path()).is_empty(), "neither result is placed");
    assert_eq!(tree(&s.copy()), before_copy, "the copy is back as it was given");
    assert_eq!(tree(s.local_root.path()), before_local, "nothing is written");
}

#[test]
fn import_refuses_a_result_whose_identity_computes_to_another_run_id() {
    let s = Scene::new("import-identity");
    let local = open(s.local_root.path());
    let p = s.copied(s.remote.sim).join("identity.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    v["identity"]["seed"] = 2.into();
    std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("computes to")), "{e}");
    assert!(result_folders(s.local_root.path()).is_empty());
}

#[test]
fn import_refuses_a_result_that_did_not_complete() {
    let s = Scene::new("import-running");
    let local = open(s.local_root.path());
    let p = s.copied(s.remote.fit).join("run.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    v["status"] = "running".into();
    std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("running")), "{e}");
    assert!(result_folders(s.local_root.path()).is_empty());
}

#[test]
fn import_refuses_a_result_no_invocation_record_in_the_copy_lists() {
    let s = Scene::new("import-unlisted");
    let local = open(s.local_root.path());
    // The record of the invocation that made the prediction.
    let invs = s.copy().join(".invocations");
    std::fs::remove_file(invs.join("2.json")).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    match &e {
        Error::ImportRefused { folder, reason } => {
            assert_eq!(folder, &s.copied(s.remote.pred));
            assert!(reason.contains("invocation record"), "{reason}");
        }
        other => panic!("{other}"),
    }
}

#[test]
fn import_of_a_result_already_here_with_equal_bytes_leaves_it_in_place() {
    let s = Scene::new("import-present");
    // The same build as the copy's, and the same simulation computed here:
    // its provenance file differs, its result files do not.
    let local = open_build(s.local_root.path(), "remote");
    let (id, attributes) = sim(1);
    let Begin::Claimed(mut run) = local.begin::<Sim>(&id, &attributes, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.expect(files(&["traj.tsv"], &["execution.json"]));
    run.result_file("traj.tsv").unwrap().write_all(b"t\tI\n0\t1\n").unwrap();
    run.provenance_file("execution.json").unwrap().write_all(b"{\"host\":\"here\"}").unwrap();
    let here = run.finish().unwrap();
    assert_eq!(here.id().key(), s.remote.sim, "premise: one run id");
    let ino = |p: &Path| std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(p).unwrap());
    let before = ino(&here.folder().join("traj.tsv"));

    let i = labelled("pulled");
    let got = local.import(&s.copy(), &i).unwrap();
    let inv = i.finish(&local).unwrap();
    assert_eq!(got.present, vec![s.remote.sim]);
    assert!(got.placed.contains(&s.remote.fit) && got.placed.contains(&s.remote.pred));
    // The obs by-product the copy holds inside the simulation is placed
    // inside the folder held here.
    assert!(got.placed.contains(&s.remote.obs));
    assert!(local.locate(s.remote.obs).unwrap().folder().starts_with(here.folder()));
    assert_eq!(ino(&here.folder().join("traj.tsv")), before, "the stored folder is left in place");
    assert_eq!(std::fs::read(here.folder().join("execution.json")).unwrap(), b"{\"host\":\"here\"}");
    assert!(inv.record.listings().iter().any(|e| e.run_id == s.remote.sim && e.outcome == Some(&crate::record::Outcome::Reused)));
    assert!(local.verify(None).unwrap().damaged.is_empty());
}

#[test]
fn import_of_a_result_here_with_other_bytes_is_refused_naming_both() {
    let s = Scene::new("import-differs");
    let local = open_build(s.local_root.path(), "remote");
    let here = fit_in(&local, &info(), b"other draws");
    assert_eq!(here.id().key(), s.remote.fit, "premise: one run id");
    let before = tree(s.local_root.path());
    let e = local.import(&s.copy(), &info()).unwrap_err();
    match &e {
        Error::ImportedDiffers { run_id, local: l, imported } => {
            assert_eq!(*run_id, s.remote.fit);
            assert_eq!(l, here.folder());
            assert_eq!(imported, &s.copied(s.remote.fit));
        }
        other => panic!("{other}"),
    }
    let after = tree(s.local_root.path());
    let new: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
    assert!(new.is_empty(), "nothing placed: {new:?}");
}

#[test]
fn a_folder_path_held_by_another_result_is_refused() {
    let s = Scene::new("import-path");
    let local = open(s.local_root.path());
    let rel = s.copied(s.remote.sim).strip_prefix(s.copy()).unwrap().to_path_buf();
    let taken = s.local_root.path().join(&rel);
    // Another result's records in the folder at the sim's path.
    let other_root = TempRoot::new("import-path-other");
    let other = run_sim(&open(other_root.path()), &info(), 7);
    std::fs::create_dir_all(taken.parent().unwrap()).unwrap();
    crate::fsutil::copy_tree(other.folder(), &taken).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportedDiffers { run_id, local: l, .. } if *run_id == s.remote.sim && l == &taken), "{e}");
    std::fs::remove_dir_all(&taken).unwrap();

    // A folder there that holds no result is refused saying so.
    std::fs::create_dir_all(&taken).unwrap();
    std::fs::write(taken.join("other.tsv"), b"x").unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("holds no result")), "{e}");
    assert_eq!(result_folders(s.local_root.path()), Vec::<String>::new());
}

/// A result folder moved to where this store would not look for it: under
/// another kind's directory, or renamed without its run id.
#[test]
fn a_result_away_from_its_layout_path_is_refused() {
    let s = Scene::new("import-misplaced");
    let local = open(s.local_root.path());
    let sim = s.copied(s.remote.sim);
    let kind_dir = sim.parent().unwrap().parent().unwrap().to_path_buf();
    let moved = kind_dir.with_file_name("simulations");
    std::fs::rename(&kind_dir, &moved).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("not at a path")), "{e}");
    std::fs::rename(&moved, &kind_dir).unwrap();
    let renamed = sim.with_file_name("seed_1");
    std::fs::rename(&sim, &renamed).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { folder, reason } if folder == &renamed && reason.contains("run id")), "{e}");
    assert!(result_folders(s.local_root.path()).is_empty());
}

/// The copy holds the prediction inside a folder at the fit's path that holds
/// nothing of the fit.
fn copy_without_the_fit(s: &Scene) {
    let fit = s.copied(s.remote.fit);
    for e in std::fs::read_dir(&fit).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::remove_file(e.path()).unwrap();
        }
    }
}

#[test]
fn a_nested_result_whose_parent_is_here_is_placed_inside_it() {
    let s = Scene::new("import-parent-here");
    copy_without_the_fit(&s);
    let local = open_build(s.local_root.path(), "remote");
    let here = fit_in(&local, &info(), b"d");
    assert_eq!(here.id().key(), s.remote.fit, "premise: one run id");
    let got = local.import(&s.copy(), &info()).unwrap();
    assert!(got.placed.contains(&s.remote.pred));
    assert!(!got.placed.contains(&s.remote.fit) && !got.present.contains(&s.remote.fit));
    assert!(local.locate(s.remote.pred).unwrap().folder().starts_with(here.folder()));
    assert!(local.verify(None).unwrap().damaged.is_empty());
}

#[test]
fn a_nested_result_whose_parent_is_nowhere_is_refused() {
    let s = Scene::new("import-parent-missing");
    copy_without_the_fit(&s);
    let local = open(s.local_root.path());
    let e = local.import(&s.copy(), &info()).unwrap_err();
    match &e {
        Error::ImportRefused { folder, reason } => {
            assert_eq!(folder, &s.copied(s.remote.pred));
            assert!(reason.contains("neither in the copy nor in this store"), "{reason}");
        }
        other => panic!("{other}"),
    }
    assert!(result_folders(s.local_root.path()).is_empty());
}

#[test]
fn a_copy_of_another_layout_is_refused() {
    let s = Scene::new("import-layout");
    let root = TempRoot::new("import-bykind");
    let local = Store::open(root.path(), Build::Known(Engine { build: "b1".into() }), TopLevel::ByKind, kinds()).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("lays results out")), "{e}");
    assert!(s.copy().exists());
}

/// An import stopped after its first rename, as a killed process stops:
/// every folder outside `.staging/` is a whole result, the one placed is
/// found, and `clean` removes what is left once its owner has stopped.
#[test]
fn an_import_stopped_midway_leaves_the_store_valid() {
    let s = Scene::new("import-crash");
    let local = open(s.local_root.path());
    let staging = s.local_root.path().join(".staging");
    std::fs::create_dir_all(&staging).unwrap();
    let staged = staging.join(crate::fsutil::import_draft_name(&dead_owner()));
    std::fs::rename(s.copy(), &staged).unwrap();
    MOVES_LEFT.with(|m| m.set(Some(1)));
    let i = labelled("pulled");
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| local.import_staged(&staged, &s.copy(), &i)));
    MOVES_LEFT.with(|m| m.set(None));
    std::mem::forget(i);
    assert!(r.is_err(), "premise: stopped");
    let placed = result_folders(s.local_root.path());
    assert!(!placed.is_empty() && placed.len() < result_folders(s.remote_root.path()).len(), "premise: midway: {placed:?}");
    assert!(local.verify(None).unwrap().damaged.is_empty());
    for f in &placed {
        let id = crate::storage::identity_of(&s.local_root.path().join(f)).unwrap();
        assert!(local.locate(id.run_id).is_ok(), "{f}");
    }
    let c = local.clean().unwrap();
    assert!(c.staging.contains(&staged), "{:?}", c.staging);
    assert_eq!(std::fs::read_dir(&staging).unwrap().count(), 0);
    assert!(local.verify(None).unwrap().damaged.is_empty());
}

/// A second import of the same result reaching its destination first: the
/// folder it placed has the same result files, so this import records the
/// result, and the by-product inside it, as present.
#[test]
fn a_destination_filled_meanwhile_with_the_same_result_is_present() {
    let s = Scene::new("import-race");
    let local = open(s.local_root.path());
    let rel = s.copied(s.remote.sim).strip_prefix(s.copy()).unwrap().to_path_buf();
    let theirs = s.remote_root.path().join(&rel);
    let dest = s.local_root.path().join(&rel);
    let target = dest.clone();
    INTERLOPER.with(|i| {
        *i.borrow_mut() = Some(Box::new(move |d: &Path| {
            if d == target && !d.exists() {
                std::fs::create_dir_all(d.parent().unwrap()).unwrap();
                crate::fsutil::copy_tree(&theirs, d).unwrap();
            }
        }))
    });
    let got = local.import(&s.copy(), &info());
    INTERLOPER.with(|i| *i.borrow_mut() = None);
    let got = got.unwrap();
    assert!(got.present.contains(&s.remote.sim) && got.present.contains(&s.remote.obs), "{got:?}");
    assert!(got.placed.contains(&s.remote.fit) && got.placed.contains(&s.remote.pred), "{got:?}");
    assert_eq!(local.locate(s.remote.sim).unwrap().folder(), dest);
    assert!(local.verify(None).unwrap().damaged.is_empty());
}

/// A destination another writer created, empty, just before the rename is
/// not replaced: the result is refused naming it.
#[test]
fn a_destination_created_meanwhile_is_not_replaced() {
    let s = Scene::new("import-race-empty");
    let local = open(s.local_root.path());
    let rel = s.copied(s.remote.sim).strip_prefix(s.copy()).unwrap().to_path_buf();
    let dest = s.local_root.path().join(&rel);
    let target = dest.clone();
    INTERLOPER.with(|i| {
        *i.borrow_mut() = Some(Box::new(move |d: &Path| {
            if d == target && !d.exists() {
                std::fs::create_dir_all(d).unwrap();
            }
        }))
    });
    let before_copy = tree(&s.copy());
    let i = labelled("pulled");
    let e = local.import(&s.copy(), &i);
    INTERLOPER.with(|i| *i.borrow_mut() = None);
    drop(i);
    let e = e.unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { reason, .. } if reason.contains("holds no result")), "{e}");
    assert_eq!(std::fs::read_dir(&dest).unwrap().count(), 0, "the folder another writer made is left as it was");
    // The fit, placed before the sim's rename was refused, is moved back.
    assert!(result_folders(s.local_root.path()).is_empty(), "{:?}", result_folders(s.local_root.path()));
    assert!(!s.local_root.path().join(".invocations").exists(), "no invocation is recorded");
    assert_eq!(tree(&s.copy()), before_copy, "the copy is back as it was given");
}

/// A placement that fails midway, here because a directory of this store
/// cannot be written, moves every folder it placed back into the copy: the
/// store and the copy are as they were, and no invocation is recorded.
#[test]
fn a_placement_that_fails_midway_is_undone() {
    use std::os::unix::fs::PermissionsExt;
    let s = Scene::new("import-undo");
    let local = open(s.local_root.path());
    // The sim's kind directory, read-only: the fit, which sorts first, is
    // placed, and the sim cannot be.
    let sim_rel = s.copied(s.remote.sim).strip_prefix(s.copy()).unwrap().to_path_buf();
    let kind_dir = s.local_root.path().join(sim_rel.iter().take(2).collect::<PathBuf>());
    std::fs::create_dir_all(&kind_dir).unwrap();
    std::fs::set_permissions(&kind_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    let dirs = |root: &Path| -> Vec<String> {
        let mut out = Vec::new();
        fn go(d: &Path, root: &Path, out: &mut Vec<String>) {
            for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
                if e.path().is_dir() {
                    out.push(crate::fsutil::rel_key(root, &e.path()));
                    go(&e.path(), root, out);
                }
            }
        }
        go(root, root, &mut out);
        out.sort();
        out
    };
    let (before_files, before_dirs, before_copy) = (tree(s.local_root.path()), dirs(s.local_root.path()), tree(&s.copy()));
    let i = labelled("pulled");
    let r = local.import(&s.copy(), &i);
    drop(i);
    std::fs::set_permissions(&kind_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(r.is_err(), "premise: the sim cannot be placed");
    let after_dirs: Vec<String> = dirs(s.local_root.path()).into_iter().filter(|d| !d.starts_with(".staging")).collect();
    assert_eq!(after_dirs, before_dirs.into_iter().filter(|d| !d.starts_with(".staging")).collect::<Vec<_>>());
    assert_eq!(tree(s.local_root.path()), before_files, "the store is as it was");
    assert_eq!(tree(&s.copy()), before_copy, "the copy is back as it was given");
    assert!(!s.local_root.path().join(".invocations").exists(), "no invocation is recorded");
}

/// Symbolic links, which a renamed copy would carry in unread, are refused
/// wherever they are.
#[test]
fn a_symbolic_link_in_the_copy_is_refused() {
    let s = Scene::new("import-symlink");
    let local = open(s.local_root.path());
    let outside = s.pull_root.path().join("identity.json");
    let ident = s.copied(s.remote.fit).join("identity.json");
    std::fs::rename(&ident, &outside).unwrap();
    std::os::unix::fs::symlink(&outside, &ident).unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { folder, reason } if folder == &ident && reason.contains("symbolic link")), "{e}");
    assert!(result_folders(s.local_root.path()).is_empty());
}

/// A dot-named directory inside a result, which no check reads below, is
/// refused even when it looks like a nested result.
#[test]
fn a_dot_named_directory_inside_a_result_is_refused() {
    let s = Scene::new("import-sneak");
    let local = open(s.local_root.path());
    let pred = s.copied(s.remote.pred);
    let sneak = pred.parent().unwrap().join(".sneak");
    crate::fsutil::copy_tree(&pred, &sneak).unwrap();
    std::fs::write(sneak.join("payload"), b"unverified").unwrap();
    let e = local.import(&s.copy(), &info()).unwrap_err();
    assert!(matches!(&e, Error::ImportRefused { folder, reason } if folder == &sneak && reason.contains("dot-named")), "{e}");
    assert!(result_folders(s.local_root.path()).is_empty());
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "score", version = 1)]
struct ScoreId {
    model: Dep<Model>,
    n: u64,
}

/// A result that depends on a compiled model without sitting inside it.
struct Score;
impl Artifact for Score {
    const KIND: &'static str = "score";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ScoreId;
    type Attributes = ();
}
impl Primary for Score {
    fn layout(id: &ScoreId, _: &()) -> Placed {
        Layout::in_kind().result(&format!("n{}", id.n))
    }
}

fn open_scoring(root: &Path, build: &str) -> Store {
    let kinds = kinds().register::<Score>();
    Store::open(root, Build::Known(Engine { build: build.into() }), TopLevel::ByBuild { label: "0.1.0".into() }, kinds).unwrap()
}

/// Stores the model in `store` under `info`.
fn put_model(store: &Store, info: &RunInfo) -> crate::Stored<Model> {
    let content = Hashed::new(ModelText("S -> I".into()));
    store.put::<Model>(content, &ModelAttributes { name: "sir".into() }, &crate::Provenance::none(), info).unwrap()
}

#[test]
fn a_result_whose_dependency_is_nowhere_is_refused() {
    let remote = TempRoot::new("import-dep-remote");
    let (score, model) = {
        let store = open_scoring(remote.path(), "remote");
        let i = labelled("scored");
        let model = put_model(&store, &i);
        let id = ScoreId { model: model.dep("model.txt").unwrap(), n: 1 };
        let Begin::Claimed(mut run) = store.begin::<Score>(&id, &(), &i, Reuse::IfCompleted).unwrap() else { panic!() };
        run.expect(expecting(&["score.tsv"]));
        run.result_file("score.tsv").unwrap().write_all(b"s").unwrap();
        let score = run.finish().unwrap();
        i.finish(&store).unwrap();
        (score.id().key(), model.id().key())
    };
    let pulled = TempRoot::new("import-dep-pull");
    let copy = pulled.path().join("copy");
    pull(remote.path(), &copy);
    let model_dir = result_folders(&copy)
        .into_iter()
        .map(|r| copy.join(r))
        .find(|d| crate::storage::identity_of(d).is_some_and(|f| f.run_id == model))
        .unwrap();
    crate::fsutil::remove_all(&model_dir).unwrap();

    let local_root = TempRoot::new("import-dep-local");
    let local = open_scoring(local_root.path(), "remote");
    let e = local.import(&copy, &info()).unwrap_err();
    match &e {
        Error::ImportRefused { reason, .. } => {
            assert!(reason.contains("dependency") && reason.contains("compiled_model") && reason.contains(&model.short12()), "{reason}");
        }
        other => panic!("{other}"),
    }
    assert!(result_folders(local_root.path()).is_empty());

    // With the model stored here, the score is imported on its own.
    assert_eq!(put_model(&local, &info()).id().key(), model, "premise: one model run id");
    let got = local.import(&copy, &info()).unwrap();
    assert_eq!(got.placed, vec![score]);
}
