//! A store can hold `surveys/` leaves an older camdl wrote, of a kind this
//! camdl does not have. No command reads one or changes it: `list` leaves it
//! out, `show` and `cat` given its directory refuse it naming the kind and the
//! command that wrote it, a run-id prefix of it matches no run, and the walks
//! of the whole store (`dev reindex`, `dev migrate`) pass over it and say why.
//!
//! The leaf here is a real record with its `kind` and `run_id` rewritten, so
//! everything but the kind parses.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    let bin = manifest().join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    Command::new(bin)
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env_remove("CAMDL_STORE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", manifest().join("../../../ocaml/_build/default/bin/camdlc.exe"))
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

const SURVEY_RUN_ID: &str = "abababababababababababababababababababababababababababababababab";

/// Copies the committed old-store root `tests/fixtures/old_store/root` (one
/// simulation leaf; its README says what it is) to `to`, with each result
/// file's modification time set to the one its record states, so the leaf
/// reads as the store wrote it. Returns the leaf's record.
fn copy_old_store(to: &Path) -> serde_json::Value {
    let from = manifest().join("tests/fixtures/old_store/root");
    let mut record = None;
    let mut stack = vec![from.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let dest = to.join(p.strip_prefix(&from).unwrap());
            if p.is_dir() {
                std::fs::create_dir_all(&dest).unwrap();
                stack.push(p);
                continue;
            }
            std::fs::copy(&p, &dest).unwrap();
            if e.file_name() == "run.json" {
                let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                record = Some((dest.parent().unwrap().to_path_buf(), r));
            }
        }
    }
    let (dir, record) = record.expect("the fixture holds one leaf");
    for (name, entry) in record["artifacts"].as_object().unwrap() {
        let (secs, nanos) = entry["mtime"].as_str().unwrap().split_once('.').unwrap();
        let at = std::time::UNIX_EPOCH
            + std::time::Duration::new(secs.parse().unwrap(), nanos.parse().unwrap());
        std::fs::File::options().write(true).open(dir.join(name)).unwrap().set_modified(at).unwrap();
    }
    record
}

/// A store holding one pfilter run, one old-store leaf from the committed
/// fixture, and one `surveys/` leaf made from that leaf's record; returns the
/// directory and the survey leaf's path relative to it.
fn store() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fixture = manifest().join("tests/fixtures/reporting");
    for f in ["sir.camdl", "cases.tsv", "if2.toml"] {
        std::fs::copy(fixture.join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fixed.toml"), "rho = 0.5\nN0 = 2000\n").unwrap();
    let o = camdl(dir, &[
        "pfilter", "sir.camdl", "--fit", "if2.toml", "--params", "fixed.toml", "--param", "beta=0.35",
        "--param", "gamma=0.12", "--particles", "50", "--seed", "1",
    ]);
    assert!(o.status.success(), "{}", text(&o));
    let mut record = copy_old_store(&dir.join("results"));
    record["kind"] = "survey".into();
    record["run_id"] = SURVEY_RUN_ID.into();
    let leaf = "results/surveys/sir-aaaaaaaa/cfg-bbbbbbbb/box-cccccccc/seed_1-dddddddd";
    std::fs::create_dir_all(dir.join(leaf)).unwrap();
    std::fs::write(dir.join(leaf).join("run.json"), serde_json::to_vec(&record).unwrap()).unwrap();
    std::fs::write(dir.join(leaf).join("landscape.tsv"), "point_id\tbeta\tloglik\n0\t0.3\t-10\n").unwrap();
    (tmp, leaf.to_string())
}

const WHY: &str = "its run.json has `kind = \"survey\"`, which `camdl survey` wrote and this camdl does \
                   not have";

#[test]
fn list_leaves_a_survey_leaf_out() {
    let (tmp, leaf) = store();
    let o = camdl(tmp.path(), &["list"]);
    let out = text(&o);
    assert!(o.status.success(), "{out}");
    assert!(out.contains("pfilters"), "the pfilter leaf is listed:\n{out}");
    assert!(!out.contains("surveys"), "the survey leaf is not:\n{out}");
    let o = camdl(tmp.path(), &["list", "--format", "json"]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(!text(&o).contains(SURVEY_RUN_ID), "{}", text(&o));
    assert!(tmp.path().join(&leaf).join("landscape.tsv").is_file());
}

#[test]
fn show_and_cat_refuse_a_survey_leaf_and_say_why() {
    let (tmp, leaf) = store();
    for verb in ["show", "cat"] {
        let o = camdl(tmp.path(), &[verb, &leaf]);
        let out = text(&o);
        assert_eq!(o.status.code(), Some(1), "{verb}:\n{out}");
        assert!(out.contains("could not read run.json at") && out.contains(WHY), "{verb}:\n{out}");
    }
    // By run-id prefix it matches no run; the walk that looked says why.
    let o = camdl(tmp.path(), &["show", &SURVEY_RUN_ID[..8]]);
    let out = text(&o);
    assert_eq!(o.status.code(), Some(1), "{out}");
    assert!(out.contains("no run matches") && out.contains(WHY), "{out}");
}

#[test]
fn the_whole_store_walks_pass_over_a_survey_leaf() {
    let (tmp, leaf) = store();
    let before = std::fs::read(tmp.path().join(&leaf).join("run.json")).unwrap();

    let o = camdl(tmp.path(), &["dev", "reindex"]);
    let out = text(&o);
    assert!(o.status.success(), "{out}");
    assert!(out.contains("warning: skipping") && out.contains(WHY), "{out}");
    assert!(out.contains("reindexed 1 run(s)"), "only the pfilter leaf is indexed:\n{out}");

    let o = camdl(tmp.path(), &["dev", "migrate", "results"]);
    let out = text(&o);
    assert!(o.status.success(), "{out}");
    assert!(out.contains(WHY), "{out}");
    assert!(out.contains("1 stale left as they are, 0 refused"), "{out}");

    assert_eq!(std::fs::read(tmp.path().join(&leaf).join("run.json")).unwrap(), before,
        "the survey leaf's record is unchanged");
    assert!(tmp.path().join(&leaf).join("landscape.tsv").is_file());
}
