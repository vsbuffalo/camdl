use super::*;

#[test]
fn the_flag_wins_then_the_variable_and_a_relative_root_joins_the_working_directory() {
    let cwd = Path::new("/work/project");
    let (flag, env) = (Some(Path::new("from_flag")), Some(Path::new("/abs/from_env")));
    assert_eq!(root_from(flag, env, cwd), Some(PathBuf::from("/work/project/from_flag")));
    assert_eq!(root_from(None, env, cwd), Some(PathBuf::from("/abs/from_env")));
    assert_eq!(root_from(None, Some(Path::new("results")), cwd), Some(PathBuf::from("/work/project/results")));
    assert_eq!(root_from(None, None, cwd), None);
    assert_eq!(root_from(Some(Path::new("")), None, cwd), None, "an empty root names none");
    assert_eq!(root_from(Some(Path::new("")), env, cwd), Some(PathBuf::from("/abs/from_env")));
}

#[test]
fn a_target_is_relative_below_a_shared_parent_and_absolute_otherwise() {
    assert!(shares_a_parent(Path::new("/data/p/camdl-store"), Path::new("/data/p/results")));
    assert!(shares_a_parent(Path::new("/data/store"), Path::new("/data/p/results")));
    assert!(!shares_a_parent(Path::new("/data/store"), Path::new("/home/me/results")), "only `/` is shared");
    assert_eq!(
        relative_path(Path::new("/data/p/results/fits/m4w"), Path::new("/data/p/camdl-store/b/fit/runs/r/draws.tsv")),
        PathBuf::from("../../../camdl-store/b/fit/runs/r/draws.tsv")
    );
    assert_eq!(relative_path(Path::new("/a/b"), Path::new("/a/b/c")), PathBuf::from("c"));
}

#[test]
fn a_link_is_named_by_its_logical_name_and_stored_extension() {
    assert_eq!(link_name("draws", "draws.tsv"), "draws.tsv");
    assert_eq!(link_name("chain_0/trace", "chain_0/trace.tsv"), "chain_0/trace.tsv");
    assert_eq!(link_name("joint", "prequential.tsv"), "joint.tsv");
    assert_eq!(link_name("streams", "prequential.streams.tsv"), "streams.tsv");
    assert_eq!(link_name("sampler_summary", "pgas_summary.json"), "sampler_summary.json");
    assert_eq!(link_name("bare", "bare"), "bare");
}

/// Every kind camdl stores has a place in the link tree, decided: a group,
/// its upstream's directory, or none.
#[test]
fn every_stored_kind_has_a_placement() {
    let kinds = crate::camdl_store::camdl_kinds();
    let unplaced: Vec<&str> = kinds.names().filter(|k| placement(k).is_none()).collect();
    assert!(unplaced.is_empty(), "kinds with no placement: {unplaced:?}");
    assert_eq!(placement("no_such_kind"), None);
}

#[test]
fn a_directory_is_ours_only_when_it_holds_what_its_record_names() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("x");
    assert!(!ours(&dir, &dir, "x").unwrap(), "nothing there");
    std::fs::create_dir(&dir).unwrap();
    assert!(ours(&dir, &dir, "x").unwrap(), "an empty directory");
    let record = Record { schema: RECORD_SCHEMA.into(), name: "x".into(), links: BTreeMap::from([("a/b.tsv".to_string(), PathBuf::from("../t.tsv"))]) };
    write_dir(tmp.path(), "x", &record).unwrap();
    assert!(ours(&dir, &dir, "x").unwrap(), "a directory camdl wrote");
    assert_eq!(std::fs::read_link(dir.join("a/b.tsv")).unwrap(), PathBuf::from("../t.tsv"));
    let leftovers: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten().map(|e| e.file_name()).collect();
    assert_eq!(leftovers, vec![std::ffi::OsString::from("x")], "no temporary directory is left");

    std::os::unix::fs::symlink("elsewhere", dir.join("a/c.tsv")).unwrap();
    assert!(ours(&dir, &dir, "x").unwrap_err().contains("a/c.tsv"), "an unrecorded link");
    std::fs::remove_file(dir.join("a/c.tsv")).unwrap();
    std::fs::remove_file(dir.join("a/b.tsv")).unwrap();
    std::os::unix::fs::symlink("../other.tsv", dir.join("a/b.tsv")).unwrap();
    assert!(ours(&dir, &dir, "x").unwrap_err().contains("a/b.tsv"), "a recorded name with another target");
    std::fs::remove_file(dir.join("a/b.tsv")).unwrap();
    std::fs::write(dir.join("a/b.tsv"), "data").unwrap();
    assert!(ours(&dir, &dir, "x").unwrap_err().contains("a file"), "a file at a recorded name");
    std::fs::remove_file(dir.join("a/b.tsv")).unwrap();
    std::fs::remove_file(dir.join(RECORD)).unwrap();
    std::fs::write(dir.join("a/notes"), "mine").unwrap();
    assert!(ours(&dir, &dir, "x").unwrap_err().contains("notes"), "a file in a directory without a record");
}

#[test]
fn a_new_directory_replaces_one_camdl_wrote_and_nothing_else() {
    let tmp = tempfile::tempdir().unwrap();
    let one = Record { schema: RECORD_SCHEMA.into(), name: "x".into(), links: BTreeMap::from([("old.tsv".to_string(), PathBuf::from("t1"))]) };
    let two = Record { schema: RECORD_SCHEMA.into(), name: "x".into(), links: BTreeMap::from([("new.tsv".to_string(), PathBuf::from("t2"))]) };
    write_dir(tmp.path(), "x", &one).unwrap();
    write_dir(tmp.path(), "x", &two).unwrap();
    let names: Vec<_> = std::fs::read_dir(tmp.path().join("x")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert_eq!(names.len(), 2, "the record and new.tsv: {names:?}");
    assert!(tmp.path().join("x/new.tsv").symlink_metadata().is_ok() && tmp.path().join("x/old.tsv").symlink_metadata().is_err());

    std::fs::write(tmp.path().join("x/mine"), "keep").unwrap();
    assert!(write_dir(tmp.path(), "x", &one).unwrap_err().contains("mine"));
    assert_eq!(std::fs::read(tmp.path().join("x/mine")).unwrap(), b"keep");
    let leftovers: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten().map(|e| e.file_name()).collect();
    assert_eq!(leftovers, vec![std::ffi::OsString::from("x")], "the refused build is removed");
}

/// A directory whose record names another label is refused, naming both:
/// what a case-insensitive volume shows `up` when `Up` wrote it. The record
/// is planted, so the check runs on any filesystem.
#[test]
fn a_directory_recorded_for_another_label_is_refused_naming_both() {
    let tmp = tempfile::tempdir().unwrap();
    let up = Record { schema: RECORD_SCHEMA.into(), name: "Up".into(), links: BTreeMap::from([("traj.tsv".to_string(), PathBuf::from("t"))]) };
    write_dir(tmp.path(), "up", &up).unwrap();
    let dir = tmp.path().join("up");
    let e = ours(&dir, &dir, "up").unwrap_err();
    assert!(e.contains("`Up`") && e.contains("`up`"), "{e}");
    assert!(ours(&dir, &dir, "Up").unwrap());
    // NFC and NFD spellings of one word are two labels.
    let (nfc, nfd) = ("caf\u{e9}", "cafe\u{301}");
    let cafe = Record { schema: RECORD_SCHEMA.into(), name: nfd.into(), links: BTreeMap::new() };
    write_dir(tmp.path(), "cafe", &cafe).unwrap();
    let e = ours(&tmp.path().join("cafe"), &tmp.path().join("cafe"), nfc).unwrap_err();
    assert!(e.contains(nfc) && e.contains(nfd), "{e}");

    // The claim refuses it, and leaves it in place.
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("sims")).unwrap();
    write_dir(&root.path().join("sims"), "up", &up).unwrap();
    let e = LinkTree::claim(root.path(), Some("up")).unwrap_err();
    assert!(e.contains("`Up`"), "{e}");
    assert!(root.path().join("sims/up/traj.tsv").symlink_metadata().is_ok(), "the refused directory is put back");
    let names: Vec<_> = std::fs::read_dir(root.path().join("sims")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert_eq!(names, vec![std::ffi::OsString::from("up")], "no directory is left aside");
}

#[test]
fn a_label_too_long_for_a_temporary_name_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let long = "a".repeat(LABEL_MAX + 1);
    let e = LinkTree::claim(root.path(), Some(&long)).unwrap_err();
    assert!(e.contains("too long"), "{e}");
    assert!(LinkTree::claim(root.path(), Some(&"a".repeat(LABEL_MAX))).is_ok());
    // The longest label's temporary name fits.
    assert!(format!(".{}.camdl-new.{}", "a".repeat(LABEL_MAX), temporary_tag()).len() <= NAME_MAX);
}

/// The temporary directories of a process that is no longer running are
/// removed at the claim; one of a running process is left; one holding a
/// file camdl did not write is refused.
#[test]
fn a_killed_commands_temporary_directories_are_removed() {
    let root = tempfile::tempdir().unwrap();
    let sims = root.path().join("sims");
    std::fs::create_dir(&sims).unwrap();
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap();
    let record = Record { schema: RECORD_SCHEMA.into(), name: "x".into(), links: BTreeMap::from([("traj.tsv".to_string(), PathBuf::from("t"))]) };
    write_dir(&sims, "x", &record).unwrap();
    std::fs::rename(sims.join("x"), sims.join(format!(".x.camdl-new.{dead}.1"))).unwrap();
    std::fs::create_dir(sims.join(format!(".x.camdl-old.{dead}.2"))).unwrap();
    let live = format!(".x.camdl-new.{}.3", std::process::id());
    std::fs::create_dir(sims.join(&live)).unwrap();
    LinkTree::claim(root.path(), Some("x")).unwrap();
    let names: Vec<String> = std::fs::read_dir(&sims).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names, vec![live], "only a running process's directory is left");

    let foreign = sims.join(format!(".x.camdl-old.{dead}.4"));
    std::fs::create_dir(&foreign).unwrap();
    std::fs::write(foreign.join("mine"), "keep").unwrap();
    let e = LinkTree::claim(root.path(), Some("x")).unwrap_err();
    assert!(e.contains("mine"), "{e}");
    assert_eq!(std::fs::read(foreign.join("mine")).unwrap(), b"keep");
}

#[test]
fn a_path_not_yet_made_resolves_through_its_existing_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("real")).unwrap();
    std::os::unix::fs::symlink(tmp.path().join("real"), tmp.path().join("via")).unwrap();
    let got = resolved_prefix(&tmp.path().join("via/a/b")).unwrap();
    assert_eq!(got, tmp.path().canonicalize().unwrap().join("real/a/b"));
}
