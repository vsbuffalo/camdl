//! A result's life: claim, write, finish or fail, reuse, retry, reclaim and
//! quarantine. These carry over the cases of the `runid` store's tests to the
//! new store's API.

use std::path::Path;
use std::time::Duration;

use crate::tests::*;
use crate::record::{InvocationFile, Outcome, RunFile, Status};
use crate::{Begin, EntryState, Error, Id, Provenance, Reuse, Wanted};

fn read_run(dir: &Path) -> RunFile {
    serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap()
}

fn sim_id(store: &Store, seed: u64) -> Id<Sim> {
    Id::of(&store.inner.ambient_key, &sim(seed).0)
}

fn fit(method: &str) -> FitId {
    FitId { problem: "sir-cases".into(), method: method.into() }
}

fn claim_fit(store: &Store, info: &RunInfo, method: &str) -> crate::Run<Fit> {
    match store.begin::<Fit>(&fit(method), &(), info, Reuse::IfCompleted).unwrap() {
        Begin::Claimed(r) => r,
        Begin::Reused(_) => panic!("reused"),
        Begin::Busy(o) => panic!("busy: {o}"),
    }
}

#[test]
fn an_absent_result_is_claimed_then_reused_and_its_folder_follows_the_layout() {
    let root = TempRoot::new("claim");
    let store = open(root.path());
    let first = run_sim(&store, &info(), 1);
    let folders = result_folders(root.path());
    assert_eq!(folders.len(), 1, "{folders:?}");
    let parts: Vec<&str> = folders[0].split('/').collect();
    let id = sim_id(&store, 1);
    assert_eq!(parts[0], format!("0.1.0-{}", store.inner.ambient_key.short12()));
    assert_eq!(parts[1], "sim");
    assert_eq!(parts[2], format!("sir-{}", "sir".to_string().content_hash().short12()));
    assert_eq!(parts[3], format!("seed_1-{}", id.key().short12()));
    assert_eq!(first.id(), &id);

    let run = read_run(first.folder());
    assert_eq!(run.status, Status::Completed);
    let entry = &run.files["traj.tsv"];
    assert_eq!(entry.sha256, Some(Key::digest_bytes(b"t\tI\n0\t1\n")));

    match store.begin::<Sim>(&sim(1).0, &sim(1).1, &info(), Reuse::IfCompleted).unwrap() {
        Begin::Reused(s) => assert_eq!(s.folder(), first.folder()),
        _ => panic!("a completed result is reused"),
    }
}

use crate::ContentAddressed;
use crate::Key;

#[test]
fn results_that_share_a_group_share_its_directory() {
    let root = TempRoot::new("group");
    let store = open(root.path());
    let a = run_sim(&store, &info(), 1);
    let b = run_sim(&store, &info(), 2);
    assert_ne!(a.folder(), b.folder());
    assert_eq!(a.folder().parent(), b.folder().parent());
}

#[test]
fn a_failed_atomic_result_leaves_no_folder_and_its_reason_is_in_the_invocation_record() {
    let root = TempRoot::new("fail-atomic");
    let store = open(root.path());
    let info = info();
    let (id, labels) = sim(3);
    let r = store.run::<Sim, String>(&id, &labels, &info, Reuse::IfCompleted, "simulating", |run| {
        run.result_file("traj.tsv").unwrap().write_all(b"partial").unwrap();
        Err("the rate went negative".into())
    });
    assert!(matches!(r, Err(crate::RunError::Compute(_))));
    assert!(result_folders(root.path()).is_empty());
    assert_eq!(std::fs::read_dir(root.path().join(".staging")).unwrap().count(), 0);
    let inv = info.finish(&store).unwrap();
    let e = &inv.file.results[0];
    assert_eq!(e.outcome, Outcome::Failed);
    let f = e.failure.as_ref().unwrap();
    assert_eq!(f.operation.as_deref(), Some("simulating"));
    assert_eq!(f.reason.as_deref(), Some("the rate went negative"));
}

#[test]
fn a_dropped_atomic_claim_leaves_nothing_and_the_next_begin_claims() {
    let root = TempRoot::new("drop-atomic");
    let store = open(root.path());
    let (id, labels) = sim(4);
    match store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() {
        Begin::Claimed(run) => drop(run),
        _ => panic!(),
    }
    assert!(result_folders(root.path()).is_empty());
    assert!(matches!(store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap(), Begin::Claimed(_)));
}

#[test]
fn an_in_place_run_is_visible_while_it_runs_and_completes() {
    let root = TempRoot::new("inplace");
    let store = open(root.path());
    let info = info();
    let run = claim_fit(&store, &info, "if2");
    run.result_file("chain_0/trace.tsv").unwrap().write_all(b"iter\tll\n").unwrap();
    let dir = result_folders(root.path());
    assert_eq!(dir.len(), 1);
    let dir = root.path().join(&dir[0]);
    assert!(dir.join(".lock").is_file());
    assert_eq!(read_run(&dir).status, Status::Running);
    let entry = store.inspect(&run.id()).unwrap().unwrap();
    assert!(matches!(entry.state(), EntryState::Running { .. }));
    // This process holds the claim, so a second begin is refused rather than waited for.
    assert!(matches!(store.begin::<Fit>(&fit("if2"), &(), &info, Reuse::IfCompleted).unwrap(), Begin::Busy(_)));
    let stored = run.finish().unwrap();
    assert!(!dir.join(".lock").exists());
    assert_eq!(read_run(&dir).status, Status::Completed);
    assert!(read_run(&dir).files.contains_key("chain_0/trace.tsv"));
    assert_eq!(stored.folder(), dir);
}

#[test]
fn a_failed_in_place_run_is_marked_failed_and_a_retry_keeps_the_attempt() {
    let root = TempRoot::new("retry");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "pmmh");
    run.result_file("draws.tsv").unwrap().write_all(b"x").unwrap();
    run.event(&serde_json::json!({"step": 1})).unwrap();
    let dir = run.core.files_dir_pub();
    run.fail("fitting", "the chain diverged").unwrap();
    let rec = read_run(&dir);
    assert_eq!(rec.status, Status::Failed);
    assert_eq!(rec.failure.as_ref().unwrap().reason.as_deref(), Some("the chain diverged"));
    assert!(!dir.join(".lock").exists());
    match store.inspect(&Id::<Fit>::of(&store.inner.ambient_key, &fit("pmmh"))).unwrap().unwrap().state() {
        EntryState::Failed { reason, .. } => assert_eq!(reason.as_deref(), Some("the chain diverged")),
        s => panic!("{s:?}"),
    }

    let run = claim_fit(&store, &info(), "pmmh");
    let attempt = dir.join(".attempts/1");
    assert!(attempt.join("attempt.json").is_file());
    assert!(attempt.join("draws.tsv").is_file(), "result files of the attempt are kept");
    assert!(attempt.join("events.jsonl").is_file());
    assert!(!dir.join("draws.tsv").exists());
    run.result_file("draws.tsv").unwrap().write_all(b"y").unwrap();
    let stored = run.finish().unwrap();
    assert_eq!(stored.result_files(), vec!["draws.tsv"]);
    assert!(store.get(stored.id()).unwrap().is_some(), "earlier attempts do not damage the result");
}

#[test]
fn retries_keep_at_most_three_attempts() {
    let root = TempRoot::new("attempts");
    let store = open(root.path());
    let mut dir = None;
    for _ in 0..5 {
        let run = claim_fit(&store, &info(), "nuts");
        dir = Some(run.core.files_dir_pub());
        run.fail("fitting", "again").unwrap();
    }
    let _ = claim_fit(&store, &info(), "nuts");
    let mut kept: Vec<String> = std::fs::read_dir(dir.unwrap().join(".attempts"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    kept.sort();
    assert_eq!(kept, vec!["3", "4", "5"]);
}

#[test]
fn a_dropped_in_place_claim_is_failed_with_no_reason() {
    let root = TempRoot::new("drop-inplace");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "mh");
    let dir = run.core.files_dir_pub();
    drop(run);
    let rec = read_run(&dir);
    assert_eq!(rec.status, Status::Failed);
    assert_eq!(rec.failure, Some(crate::record::Failure { operation: None, reason: None }));
    assert!(!dir.join(".lock").exists());
}

#[test]
fn an_in_place_run_whose_owner_died_reads_as_interrupted_and_is_reclaimed() {
    let root = TempRoot::new("interrupted");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "if2");
    run.result_file("chain_0/trace.tsv").unwrap().write_all(b"1").unwrap();
    let dir = run.core.files_dir_pub();
    let id = run.id();
    std::mem::forget(run);
    let dead = dead_owner();
    set_owner(&dir.join("run.json"), &dead);
    set_owner(&dir.join(".lock"), &dead);
    let before = std::fs::read(dir.join("run.json")).unwrap();
    match store.inspect(&id).unwrap().unwrap().state() {
        EntryState::Interrupted { owner } => assert_eq!(owner, &dead),
        s => panic!("{s:?}"),
    }
    assert_eq!(std::fs::read(dir.join("run.json")).unwrap(), before, "a reader does not rewrite the record");

    let run = claim_fit(&store, &info(), "if2");
    assert!(dir.join(".attempts/1/attempt.json").is_file());
    assert!(!dir.join("chain_0").exists(), "subdirectories are cleared");
    run.finish().unwrap();
}

#[test]
fn a_run_held_from_another_host_is_busy_and_force_moves_it_to_quarantine() {
    let root = TempRoot::new("remote");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "if2");
    let dir = run.core.files_dir_pub();
    std::mem::forget(run);
    let remote = remote_owner();
    set_owner(&dir.join("run.json"), &remote);
    set_owner(&dir.join(".lock"), &remote);
    match store.begin::<Fit>(&fit("if2"), &(), &info(), Reuse::IfCompleted).unwrap() {
        Begin::Busy(o) => assert_eq!(o, remote),
        _ => panic!("a run held from another host is not reclaimed"),
    }
    match store.begin::<Fit>(&fit("if2"), &(), &info(), Reuse::Force).unwrap() {
        Begin::Claimed(run) => {
            run.finish().unwrap();
        }
        _ => panic!("force reclaims a run that stopped on another host"),
    }
    assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1);
}

#[test]
fn an_atomic_claim_held_by_a_live_process_is_waited_for() {
    let root = TempRoot::new("wait");
    let store = open(root.path());
    let (id, labels) = sim(5);
    let staging = match store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() {
        Begin::Claimed(run) => {
            let d = run.core.files_dir_pub();
            std::mem::forget(run);
            d
        }
        _ => panic!(),
    };
    let (owner, mut child) = live_owner();
    set_owner(&staging.join("run.json"), &owner);
    let killer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2500));
        child.kill().unwrap();
        child.wait().unwrap();
    });
    let t0 = std::time::Instant::now();
    let b = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap();
    killer.join().unwrap();
    assert!(t0.elapsed() >= Duration::from_millis(2000), "the claim was waited for");
    assert!(matches!(b, Begin::Claimed(_)), "a claim whose owner died is taken");
}

#[test]
fn an_atomic_claim_whose_owner_is_dead_is_removed_and_claimed() {
    let root = TempRoot::new("dead-atomic");
    let store = open(root.path());
    let (id, labels) = sim(6);
    let staging = match store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() {
        Begin::Claimed(run) => {
            let d = run.core.files_dir_pub();
            std::mem::forget(run);
            d
        }
        _ => panic!(),
    };
    set_owner(&staging.join("run.json"), &dead_owner());
    assert!(matches!(store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap(), Begin::Claimed(_)));
}

#[test]
fn a_damaged_result_is_not_served_and_the_next_begin_quarantines_it() {
    for (tag, damage) in [
        ("changed", Box::new(|d: &Path| std::fs::write(d.join("traj.tsv"), b"changed bytes").unwrap()) as Box<dyn Fn(&Path)>),
        ("missing", Box::new(|d: &Path| std::fs::remove_file(d.join("traj.tsv")).unwrap())),
        ("unlisted", Box::new(|d: &Path| std::fs::write(d.join("stray.tsv"), b"x").unwrap())),
        ("format", Box::new(|d: &Path| {
            let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("run.json")).unwrap()).unwrap();
            v["format"] = 0.into();
            std::fs::write(d.join("run.json"), serde_json::to_vec(&v).unwrap()).unwrap();
        })),
    ] {
        let root = TempRoot::new(tag);
        let store = open(root.path());
        let s = run_sim(&store, &info(), 7);
        damage(s.folder());
        assert!(store.get(s.id()).unwrap().is_none(), "{tag}: a damaged result is not served");
        assert!(matches!(store.inspect(s.id()).unwrap().unwrap().state(), EntryState::Damaged { .. }), "{tag}");
        let again = run_sim(&store, &info(), 7);
        assert_eq!(again.id(), s.id());
        assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1, "{tag}");
    }
}

#[test]
fn force_quarantines_a_completed_result_and_recomputes() {
    let root = TempRoot::new("force");
    let store = open(root.path());
    let s = run_sim(&store, &info(), 8);
    let (id, labels) = sim(8);
    match store.begin::<Sim>(&id, &labels, &info(), Reuse::Force).unwrap() {
        Begin::Claimed(run) => {
            run.result_file("traj.tsv").unwrap().write_all(b"t\tI\n0\t8\n").unwrap();
            run.finish().unwrap();
        }
        _ => panic!("force recomputes"),
    }
    assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1);
    assert!(store.get(s.id()).unwrap().is_some());
}

#[test]
fn an_incomplete_build_never_reuses() {
    let root = TempRoot::new("incomplete");
    let store = Store::open(root.path(), Build::Incomplete(Engine { build: "?".into() }), TopLevel::ByBuild { label: "x".into() }, kinds()).unwrap();
    run_sim(&store, &info(), 9);
    let (id, labels) = sim(9);
    assert!(matches!(store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap(), Begin::Claimed(_)));
}

#[test]
fn progress_and_events_are_attempt_files_outside_the_manifest() {
    let root = TempRoot::new("progress");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "if2");
    run.progress(&serde_json::json!({"iteration": 3})).unwrap();
    run.event(&"started").unwrap();
    run.event(&"halfway").unwrap();
    run.attempt_file("execution.json").unwrap().write_all(b"{}").unwrap();
    let id = run.id();
    let entry = store.inspect(&id).unwrap().unwrap();
    assert_eq!(entry.progress().unwrap(), Some(serde_json::json!({"iteration": 3})));
    let stored = run.finish().unwrap();
    let rec = read_run(stored.folder());
    for f in ["progress.json", "events.jsonl", "execution.json"] {
        assert_eq!(rec.files[f].role, crate::record::Role::Attempt, "{f}");
        assert_eq!(rec.files[f].sha256, None);
    }
    let events = std::fs::read_to_string(stored.folder().join("events.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = events.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines[1]["n"], 2);
    assert_eq!(lines[1]["event"], "halfway");
    assert!(store.get(&id).unwrap().is_some());
}

#[test]
fn a_folder_name_taken_by_another_result_gets_the_whole_run_id() {
    let root = TempRoot::new("collision");
    let store = open(root.path());
    let other = run_sim(&store, &info(), 11);
    let id = sim_id(&store, 12);
    let taken = other.folder().parent().unwrap().join(format!("seed_12-{}", id.key().short12()));
    std::fs::rename(other.folder(), &taken).unwrap();
    let s = run_sim(&store, &info(), 12);
    assert_eq!(s.folder().file_name().unwrap().to_string_lossy(), format!("seed_12-{}", id.key().to_hex()));
    assert!(store.get(&id).unwrap().is_some());
    let moved = store.get(other.id()).unwrap().unwrap();
    assert_eq!(moved.folder(), taken, "a result is found by its record, under any name");
}

#[test]
fn file_names_are_checked() {
    let root = TempRoot::new("names");
    let store = open(root.path());
    let (id, labels) = sim(13);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    for bad in ["../x", "/abs", ".hidden", "a/.b", "run.json", "identity.json", "progress.json", "obs/x", "a:b", "a,b", "*", ""] {
        assert!(matches!(run.result_file(bad), Err(Error::BadFileName { .. })), "{bad:?}");
    }
    let h = run.result_file("traj.tsv").unwrap();
    assert!(matches!(run.provenance_file("traj.tsv"), Err(Error::FileOpenedTwice { .. })));
    let h2 = h.clone();
    std::thread::spawn(move || h2.write_all(b"from a thread\n").unwrap()).join().unwrap();
    {
        let mut w = h.writer();
        std::io::Write::write_all(&mut w, b"buffered\n").unwrap();
    }
    let s = run.finish().unwrap();
    assert!(matches!(h.write_all(b"late"), Err(Error::WriteAfterEnd { .. })));
    let mut text = String::new();
    std::io::Read::read_to_string(&mut s.open("traj.tsv").unwrap(), &mut text).unwrap();
    assert_eq!(text, "from a thread\nbuffered\n");
}

#[derive(Identity, Clone, Debug)]
#[identity(tag = "table", version = 1)]
struct TableId {
    rows: Hashed<ModelText>,
}

struct Table;
impl Artifact for Table {
    const KIND: &'static str = "table";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = TableId;
    type Labels = ();
}
impl Primary for Table {
    fn layout(_: &TableId, _: &()) -> Placed {
        Layout::in_kind().result("table")
    }
}

#[test]
fn finish_refuses_held_content_with_no_result_file_of_its_digest() {
    let root = TempRoot::new("held");
    let store = open(root.path());
    let id = TableId { rows: Hashed::new(ModelText("a,b\n1,2\n".into())) };
    let Begin::Claimed(run) = store.begin::<Table>(&id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.result_file("rows.csv").unwrap().write_all(b"other bytes").unwrap();
    assert!(matches!(run.finish(), Err(Error::ContentMissing { .. })));
    let Begin::Claimed(run) = store.begin::<Table>(&id, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.result_file("rows.csv").unwrap().write_all(b"a,b\n1,2\n").unwrap();
    let s = run.finish().unwrap();
    assert_eq!(s.identity().rows, id.rows.digest(), "the record holds the content's digest");
}

/// Openers racing to create a root that does not exist yet all open it, in
/// every one of many trials: exactly one writes `store.json` and the others
/// read it.
#[test]
fn concurrent_openers_of_a_root_that_does_not_exist_all_open_it() {
    for trial in 0..40 {
        let parent = TempRoot::new("open-race-fresh");
        let root = parent.path().join(format!("store-{trial}"));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let root = root.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    Store::open(&root, Build::Known(Engine { build: "b".into() }), TopLevel::ByKind, kinds()).err().map(|e| e.to_string())
                })
            })
            .collect();
        let refused: Vec<String> = handles.into_iter().filter_map(|h| h.join().unwrap()).collect();
        assert!(refused.is_empty(), "trial {trial}: {refused:?}");
    }
}

/// A foreign non-empty directory, with no sign of another opener, is refused
/// at once.
#[test]
fn a_foreign_directory_is_refused_without_waiting() {
    let root = TempRoot::new("open-foreign");
    std::fs::write(root.path().join("notes.txt"), b"x").unwrap();
    let t = std::time::Instant::now();
    let r = Store::open(root.path(), Build::Known(Engine { build: "b".into() }), TopLevel::ByKind, kinds());
    assert!(matches!(r, Err(Error::NotAStore { .. })));
    assert!(t.elapsed() < std::time::Duration::from_millis(200), "refused after {:?}", t.elapsed());
}

#[test]
fn open_refuses_a_foreign_directory_the_other_top_level_and_a_kind_registered_twice() {
    let root = TempRoot::new("open");
    std::fs::write(root.path().join("notes.txt"), b"x").unwrap();
    assert!(matches!(
        Store::open(root.path(), Build::Known(Engine { build: "b".into() }), TopLevel::ByKind, kinds()),
        Err(Error::NotAStore { .. })
    ));
    let root = TempRoot::new("open2");
    open(root.path());
    assert!(matches!(
        Store::open(root.path(), Build::Known(Engine { build: "b".into() }), TopLevel::ByKind, kinds()),
        Err(Error::TopLevelDiffers { .. })
    ));
    std::fs::write(root.path().join("store.json"), br#"{"format":99,"top_level":"by_build"}"#).unwrap();
    assert!(matches!(
        Store::open(root.path(), Build::Known(Engine { build: "b".into() }), TopLevel::ByBuild { label: "x".into() }, kinds()),
        Err(Error::NotAStore { .. })
    ));
    let root = TempRoot::new("open3");
    assert!(matches!(
        Store::open(root.path(), Build::Known(Engine { build: "b".into() }), TopLevel::ByKind, kinds().register::<Sim>()),
        Err(Error::KindRegisteredTwice { kind: "sim" })
    ));
}

#[test]
fn the_invocation_record_is_written_at_the_first_call_and_when_the_command_ends() {
    let root = TempRoot::new("invocation");
    let store = open(root.path());
    let info = labelled("baseline");
    let (id, labels) = sim(14);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info, Reuse::IfCompleted).unwrap() else { panic!() };
    let path = root.path().join(".invocations/1.json");
    let first: InvocationFile = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(first.status, crate::record::InvocationStatus::Running);
    assert_eq!(first.results[0].outcome, Outcome::Planned);
    assert_eq!(first.results[0].labels, serde_json::json!({"model": "sir", "seed": 14}));
    run.result_file("traj.tsv").unwrap().write_all(b"x").unwrap();
    run.finish().unwrap();
    run_sim(&store, &info, 14);
    let inv = info.finish(&store).unwrap();
    assert_eq!(inv.seq(), 1);
    let last: InvocationFile = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(last.status, crate::record::InvocationStatus::Ended);
    assert_eq!(last.label.as_deref(), Some("baseline"));
    let outcomes: Vec<Outcome> = last.results.iter().map(|r| r.outcome).collect();
    assert_eq!(outcomes, vec![Outcome::Computed, Outcome::Reused]);
    assert_eq!(last.counts.completed, 2);
    let stored = store.get(&Id::<Sim>::of(&store.inner.ambient_key, &id)).unwrap().unwrap();
    assert_eq!(read_run(stored.folder()).invocation, Some(1));
}

/// A plan with by-products: a stored result whose wanted by-product is not
/// stored is told apart from one that holds it, and from one not stored.
#[test]
fn a_plan_with_by_products_names_a_result_that_lacks_one() {
    let root = TempRoot::new("plan-with");
    let store = open(root.path());
    run_sim(&store, &info(), 31);
    let (id, labels) = sim(32);
    let wanted = Wanted::<Sim>::none().with::<Obs>(ObsId { sim: id.record(), rate: Finite::new(0.3).unwrap() }, ());
    let i = info();
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &i, Reuse::IfCompleted).unwrap() else { panic!() };
    run.result_file("traj.tsv").unwrap().write_all(b"t\tI\n0\t32\n").unwrap();
    run.also::<Obs>().unwrap().unwrap().result_file("obs.tsv").unwrap().write_all(b"o").unwrap();
    run.finish().unwrap();
    let want = |seed: u64| {
        let (id, labels) = sim(seed);
        let w = Wanted::<Sim>::none().with::<Obs>(ObsId { sim: id.record(), rate: Finite::new(0.3).unwrap() }, ());
        (id, labels, w)
    };
    let states: Vec<crate::PlanState> = store.plan_with::<Sim>(&[want(31), want(32), want(33)]).unwrap().into_iter().map(|p| p.state).collect();
    assert_eq!(states, vec![crate::PlanState::ByProductsMissing, crate::PlanState::Completed, crate::PlanState::Absent]);
}

#[test]
fn a_batch_claims_a_repeated_identity_once_and_answers_in_the_order_asked() {
    let root = TempRoot::new("batch");
    let store = open(root.path());
    run_sim(&store, &info(), 2);
    let info = info();
    let members: Vec<(SimId, SimLabels, Wanted<Sim>)> =
        [3u64, 2, 3, 1].iter().map(|s| { let (i, l) = sim(*s); (i, l, Wanted::none()) }).collect();
    let plan = store.plan::<Sim>(&members.iter().map(|(i, l, _)| (i.clone(), l.clone())).collect::<Vec<_>>()).unwrap();
    let states: Vec<crate::PlanState> = plan.into_iter().map(|p| p.state).collect();
    assert_eq!(states, vec![crate::PlanState::Absent, crate::PlanState::Completed, crate::PlanState::Absent, crate::PlanState::Absent]);
    let mut batch = store.begin_many::<Sim>(&members, &info, Reuse::IfCompleted).unwrap();
    let mut seen = Vec::new();
    for b in batch.members() {
        match b {
            Begin::Claimed(run) => {
                seen.push(("claimed", run.identity().seed));
                run.result_file("traj.tsv").unwrap().write_all(b"x").unwrap();
                run.finish().unwrap();
            }
            Begin::Reused(s) => seen.push(("reused", s.identity().seed)),
            Begin::Busy(_) => panic!(),
        }
    }
    assert_eq!(seen, vec![("claimed", 3), ("reused", 2), ("claimed", 1)]);
    batch.progress(&serde_json::json!({"done": 3})).unwrap();
    let inv = info.finish(&store).unwrap();
    assert_eq!(inv.file.results.len(), 4, "a result asked for twice is listed twice");
    assert_eq!(inv.file.results[0].run_id, inv.file.results[2].run_id);
    assert_eq!(inv.file.results[2].outcome, Outcome::Computed);
    assert_eq!(inv.file.progress, Some(serde_json::json!({"done": 3})));
}

#[test]
fn a_content_artifact_is_stored_once_and_read_through_a_dependency() {
    let root = TempRoot::new("put");
    let store = open(root.path());
    let content = Hashed::new(ModelText("S -> I".into()));
    let prov = Provenance::none().file("source.camdl", b"model sir".to_vec()).value("whole_digest", &"abc").unwrap();
    assert!(store.get_content::<Model>(&content).unwrap().is_none(), "a content not yet put is not found");
    let info1 = info();
    let a = store.put::<Model>(content.clone(), &ModelLabels { name: "sir".into() }, &prov, &info1).unwrap();
    let b = store.put::<Model>(content.clone(), &ModelLabels { name: "renamed".into() }, &Provenance::none(), &info()).unwrap();
    assert_eq!(a.folder(), b.folder(), "the first writer's labels name the folder");
    assert_eq!(store.get_content::<Model>(&content).unwrap().unwrap().folder(), a.folder(), "a put content is found by its content");
    assert!(a.folder().file_name().unwrap().to_string_lossy().starts_with("sir-"));
    assert_eq!(a.recorded("whole_digest"), Some(&serde_json::json!("abc")));
    assert_eq!(*a.identity(), content.digest());
    let inv = info1.finish(&store).unwrap();
    assert_eq!(inv.file.results[0].outcome, Outcome::Computed);

    // A result depending on the model reads it, checked, and records the read.
    #[derive(Identity, Clone, Debug)]
    #[identity(tag = "uses_model", version = 1)]
    struct UsesId {
        model: Dep<Model>,
    }
    struct Uses;
    impl Artifact for Uses {
        const KIND: &'static str = "uses";
        const WRITE: WriteMode = WriteMode::Atomic;
        type Identity = UsesId;
        type Labels = ();
    }
    impl Primary for Uses {
        fn layout(_: &UsesId, _: &()) -> Placed {
            Layout::in_kind().result("uses")
        }
    }
    let uses = UsesId { model: a.dep(Model::FILE).unwrap() };
    let Begin::Claimed(mut run) = store.begin::<Uses>(&uses, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let model = run.load(&uses.model).unwrap();
    assert_eq!(model.0, "S -> I");
    let s = run.finish().unwrap();
    let rec = read_run(s.folder());
    assert_eq!(rec.deps.len(), 1);
    assert_eq!(rec.deps[0].kind, "compiled_model");
    assert_eq!(rec.reads.len(), 1);
}

#[test]
fn a_dependency_reads_only_the_bytes_it_names() {
    let root = TempRoot::new("deps");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "if2");
    run.result_file("draws.tsv").unwrap().write_all(b"a\n1\n").unwrap();
    run.result_file("fit_state.toml").unwrap().write_all(b"[start_values]\nbeta = 1\n[other]\nx = 2\n").unwrap();
    run.provenance_file("fit.toml").unwrap().write_all(b"# config").unwrap();
    let fit = run.finish().unwrap();

    assert!(fit.dep("fit.toml").is_err(), "a provenance file is never a dependency");
    let all = fit.dep_all();
    assert_eq!(all.file(), "*");
    let two = fit.dep_files(&["fit_state.toml", "draws.tsv"]).unwrap();
    assert_eq!(two.file(), "draws.tsv,fit_state.toml");
    assert_eq!(two.digest(), all.digest(), "the two result files are all of them");
    let part = fit
        .dep_part("fit_state.toml", "start_values", |b| Ok(String::from_utf8_lossy(b).lines().take(2).collect::<Vec<_>>().join("\n").into_bytes()))
        .unwrap();
    assert_eq!(part.file(), "fit_state.toml:start_values");

    let pred = PredId { fit: fit.dep("draws.tsv").unwrap(), n: 3 };
    let Begin::Claimed(mut run) = store.begin::<Pred>(&pred, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let mut text = String::new();
    std::io::Read::read_to_string(&mut run.open(&pred.fit).unwrap(), &mut text).unwrap();
    assert_eq!(text, "a\n1\n");
    assert!(run.open(&all).is_err(), "a dependency on several files is read with open_in");
    let mut t2 = String::new();
    std::io::Read::read_to_string(&mut run.open_in(&all, "fit_state.toml").unwrap(), &mut t2).unwrap();
    assert!(t2.starts_with("[start_values]"));
    assert!(run.open_in(&all, "fit.toml").is_err());
    let p = run.finish().unwrap();
    assert!(p.folder().starts_with(fit.folder().join("predictions")), "placed inside its parent");
    assert!(store.get(fit.id()).unwrap().is_some(), "a nested result does not damage its parent");
    let back: serde_json::Value = serde_json::from_slice(&std::fs::read(p.folder().join("identity.json")).unwrap()).unwrap();
    assert_eq!(back["parent"]["relation"], "inside");
    assert_eq!(p.identity().fit, pred.fit, "the dependency reads back from the record");
}

#[test]
fn a_result_of_another_build_is_found_and_depended_on() {
    let root = TempRoot::new("builds");
    let old = open_build(root.path(), "b1");
    let run = claim_fit(&old, &info(), "if2");
    run.result_file("draws.tsv").unwrap().write_all(b"d").unwrap();
    let fit = run.finish().unwrap();
    drop(old);
    let new = open_build(root.path(), "b2");
    assert!(new.get(fit.id()).unwrap().is_some());
    let pred = PredId { fit: new.get(fit.id()).unwrap().unwrap().dep("draws.tsv").unwrap(), n: 1 };
    assert_ne!(pred.fit.build(), new.inner.ambient_key.to_hex());
    let Begin::Claimed(run) = new.begin::<Pred>(&pred, &(), &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let p = run.finish().unwrap();
    assert!(p.folder().starts_with(fit.folder()), "a nested result follows its parent's build");
    let tops: Vec<String> = std::fs::read_dir(root.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| !n.starts_with('.') && n != "store.json").collect();
    assert_eq!(tops.len(), 1, "the new build made no directory of its own: {tops:?}");
}

#[test]
fn a_reused_process_id_with_another_start_time_is_not_read_as_the_owner() {
    let root = TempRoot::new("pid-reuse");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "if2");
    let dir = run.core.files_dir_pub();
    std::mem::forget(run);
    let (mut owner, mut child) = live_owner();
    owner.started = crate::Timestamp("2001-01-01T00:00:00.000Z".into());
    set_owner(&dir.join("run.json"), &owner);
    set_owner(&dir.join(".lock"), &owner);
    let b = store.begin::<Fit>(&fit("if2"), &(), &info(), Reuse::IfCompleted).unwrap();
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(matches!(b, Begin::Claimed(_)), "a live process with another start time is not the owner");
}

#[test]
fn an_empty_in_place_folder_is_claimed_and_debris_beside_no_record_is_quarantined() {
    let root = TempRoot::new("half-made");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "empty");
    let dir = run.core.files_dir_pub();
    run.fail("x", "y").unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    let run = claim_fit(&store, &info(), "empty");
    assert_eq!(run.core.files_dir_pub(), dir, "an empty folder is a claim not yet made, and is claimed");
    assert!(!root.path().join(".quarantine").exists());
    run.fail("x", "y").unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("stray.tsv"), b"x").unwrap();
    claim_fit(&store, &info(), "empty").finish().unwrap();
    assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1, "debris is quarantined");
}

#[test]
fn finish_refuses_while_a_buffered_writer_is_open() {
    use std::io::Write as _;
    let root = TempRoot::new("writer-open");
    let store = open(root.path());
    let (id, labels) = sim(21);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let fh = run.result_file("traj.tsv").unwrap();
    let mut w = fh.writer();
    w.write_all(b"t\tI\n0\t1\n").unwrap();
    assert!(matches!(run.finish(), Err(Error::WriterOpen { .. })), "a result is never stored without bytes still in a buffer");
    drop(w);
    assert!(store.get(&sim_id(&store, 21)).unwrap().is_none());
}

#[test]
fn an_in_place_folder_whose_identity_file_does_not_read_is_quarantined() {
    let root = TempRoot::new("bad-identity");
    let store = open(root.path());
    let run = claim_fit(&store, &info(), "m");
    let dir = run.core.files_dir_pub();
    run.finish().unwrap();
    std::fs::write(dir.join("identity.json"), b"{ truncated").unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let root2 = root.path().to_path_buf();
    std::thread::spawn(move || {
        let store = open(&root2);
        let b = store.begin::<Fit>(&fit("m"), &(), &info(), Reuse::IfCompleted).map(|b| matches!(b, Begin::Claimed(_)));
        let _ = tx.send(b.map_err(|e| e.to_string()));
    });
    let r = rx.recv_timeout(Duration::from_secs(10)).expect("begin returns");
    assert_eq!(r, Ok(true));
    assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1);
}

#[test]
fn a_label_check_sees_an_invocation_that_has_taken_its_number() {
    let root = TempRoot::new("label-window");
    let store = open(root.path());
    let i = labelled("base");
    let (id, labels) = sim(22);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &i, Reuse::IfCompleted).unwrap() else { panic!() };
    // Under the store-wide lock the record is written before the lock is
    // released, so the file holds a running record, never a placeholder.
    let first: InvocationFile = serde_json::from_slice(&std::fs::read(root.path().join(".invocations/1.json")).unwrap()).unwrap();
    assert_eq!(first.status, crate::record::InvocationStatus::Running);
    drop(run);
    std::fs::write(root.path().join(".invocations/5.json"), b"{}").unwrap();
    let (id, labels) = sim(23);
    assert!(store.begin::<Sim>(&id, &labels, &labelled("other"), Reuse::IfCompleted).is_ok());
}

#[test]
fn an_atomic_finish_that_meets_a_stored_copy_of_its_id_keeps_the_stored_one_and_says_so() {
    let root = TempRoot::new("same-id");
    let other = TempRoot::new("same-id-b");
    let store = open(root.path());
    let (id, labels) = sim(24);
    let obs = ObsId { sim: id.record(), rate: Finite::new(0.3).unwrap() };
    let wanted = Wanted::<Sim>::none().with::<Obs>(obs.clone(), ());
    let info = info();
    let Begin::Claimed(mut run) = store.begin_with(&id, &labels, &wanted, &info, Reuse::IfCompleted).unwrap() else { panic!() };
    run.result_file("traj.tsv").unwrap().write_all(b"t\tI\n0\t24\n").unwrap();
    run.also::<Obs>().unwrap().unwrap().result_file("obs.tsv").unwrap().write_all(b"o").unwrap();
    let rel = { let st = open(other.path()); run_sim(&st, &crate::tests::info(), 24).folder().strip_prefix(other.path()).unwrap().to_path_buf() };
    let dest = root.path().join(&rel);
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    assert!(std::process::Command::new("cp").arg("-Rp").arg(other.path().join(&rel)).arg(&dest).status().unwrap().success());
    let s = run.finish().unwrap();
    assert_eq!(s.folder(), dest);
    let inv = info.finish(&store).unwrap();
    let outcomes: Vec<_> = inv.file.results.iter().map(|r| (r.kind.as_str(), r.outcome)).collect();
    assert_eq!(outcomes, vec![("sim", Outcome::Reused), ("obs", Outcome::Failed)], "a by-product not stored is not called computed");
}

#[test]
fn an_atomic_finish_that_meets_a_different_stored_copy_of_its_id_is_divergent() {
    let root = TempRoot::new("same-id-div");
    let other = TempRoot::new("same-id-div-b");
    let store = open(root.path());
    let (id, labels) = sim(25);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    run.result_file("traj.tsv").unwrap().write_all(b"different bytes").unwrap();
    let rel = { let st = open(other.path()); run_sim(&st, &crate::tests::info(), 25).folder().strip_prefix(other.path()).unwrap().to_path_buf() };
    let dest = root.path().join(&rel);
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    assert!(std::process::Command::new("cp").arg("-Rp").arg(other.path().join(&rel)).arg(&dest).status().unwrap().success());
    assert!(matches!(run.finish(), Err(Error::Divergent { .. })));
    assert_eq!(std::fs::read_dir(root.path().join(".quarantine")).unwrap().count(), 1, "the recomputed bytes are kept as evidence");
}

#[test]
fn a_claim_is_cleared_only_while_it_still_names_the_owner_that_was_read() {
    let root = TempRoot::new("clear-claim");
    let store = open(root.path());
    let (id, labels) = sim(26);
    let Begin::Claimed(run) = store.begin::<Sim>(&id, &labels, &info(), Reuse::IfCompleted).unwrap() else { panic!() };
    let staging = run.core.files_dir_pub();
    // A claimant read the folder when a dead owner held it; it now names this process.
    assert!(!store.inner.clear_claim(&staging, &dead_owner(), false).unwrap());
    assert!(staging.join("run.json").is_file(), "the live claim is untouched");
    run.result_file("traj.tsv").unwrap().write_all(b"x").unwrap();
    run.finish().unwrap();
}
