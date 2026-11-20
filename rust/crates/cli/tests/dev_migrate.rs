//! `camdl dev migrate`, end to end through the release binary.
//!
//! A store at layout 0 is made by a fit run of this build, rewritten into
//! the shape an older camdl left. Three shapes reached users, and each has
//! a generator here, re-run by every test rather than committed as files:
//!
//! - **main, no command-line problem flag** ([`to_layout_zero`]): no
//!   `store_layout` field in any record and no `fit.resolved.toml`;
//!   `fit.toml.original` is the file as read, and `fit_toml_hash` its hash.
//! - **main, `--allow-forcing-extrapolation`** ([`as_main_folded_it`]): as
//!   above, except that main (from c60d0401c until the resolved config) stored
//!   `fit.toml.original` with the flag folded in, the file re-serialised with
//!   `[config] allow_forcing_extrapolation = true`, while `fit_toml_hash`
//!   hashes the file as read. The stored file is then not the file the fit
//!   read, and nothing in the folder says what that file was.
//! - **a build of the resolved-config branch** ([`without_layout_numbers`]):
//!   `fit.resolved.toml` written, and no `store_layout` field.
//!
//! What is pinned:
//!
//! 1. A reader of a stored fit's config refuses a layout-0 folder naming
//!    `camdl dev migrate <folder>`; after the migration the reader gives the
//!    output it gave the fresh folder, and the migrated `fit.resolved.toml`
//!    is the fit's own text under a header that says a migration wrote it,
//!    for a sweep point and a `from_file` `[fixed]` alike. A second run
//!    changes nothing.
//! 2. `--dry-run` lists the folders and writes nothing.
//! 3. A folder whose records disagree is refused, left as it was, and the
//!    command exits 1, while the other folders are migrated.
//! 4. A folder written by a newer camdl (a higher layout) is refused, by the
//!    reader and by the migration.
//! 5. A `fit.resolved.toml` already in a layout-0 folder (a branch build's,
//!    here one made under `--allow-forcing-extrapolation`, or one a migration
//!    wrote before it stopped) is kept when it agrees with the records, and
//!    refuses the folder when it does not.
//! 6. A folder main wrote under `--allow-forcing-extrapolation` is refused,
//!    saying why its fit file cannot be checked, not that it was edited.

mod stored_fit_fixture;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use stored_fit_fixture::*;

/// Remove `store_layout` from every `run.json` and `fit.meta.json` under
/// `results`: a store a build of the resolved-config branch wrote.
fn without_layout_numbers(results: &Path) {
    for record in named(results, "run.json").into_iter().chain(named(results, "fit.meta.json")) {
        let mut v = json(&record);
        assert_eq!(v["store_layout"], 1, "{} is written at layout 1", record.display());
        v.as_object_mut().unwrap().remove("store_layout");
        set_json(&record, &v);
    }
}

/// Rewrite the store at `results` to what main left for a fit with no
/// command-line problem flag: no `store_layout` in any record, and no
/// `fit.resolved.toml`. Returns each fit folder's `fit.resolved.toml` as the
/// fit wrote it.
fn to_layout_zero(results: &Path) -> BTreeMap<PathBuf, String> {
    let mut resolved = BTreeMap::new();
    without_layout_numbers(results);
    for file in named(results, "fit.resolved.toml") {
        resolved.insert(file.parent().unwrap().to_path_buf(), std::fs::read_to_string(&file).unwrap());
        std::fs::remove_file(&file).unwrap();
    }
    resolved
}

/// [`to_layout_zero`], then each `fit.toml.original` folded as main stored it
/// for a fit run with `--allow-forcing-extrapolation`: the file parsed, the
/// flag set, the table written back (`ProblemFlags::fold_into` as of
/// c60d0401c), leaving `fit_toml_hash` the hash of the file as read.
fn as_main_folded_it(results: &Path) {
    to_layout_zero(results);
    for original in named(results, "fit.toml.original") {
        let mut doc: toml::Table = std::fs::read_to_string(&original).unwrap().parse().unwrap();
        doc.entry("config")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .unwrap()
            .insert("allow_forcing_extrapolation".to_string(), toml::Value::Boolean(true));
        std::fs::write(&original, toml::to_string(&doc).unwrap()).unwrap();
    }
}

/// The text below a `fit.resolved.toml`'s opening comment.
fn body(text: &str) -> String {
    text.lines().skip_while(|l| l.starts_with('#')).collect::<Vec<_>>().join("\n")
}

/// A sweep over a `from_file` parameter, migrated from layout 0: the readers refuse it naming the migration, a dry
/// run writes nothing, and after the migration `fit predict` gives what it
/// gave the fresh folders, from the fit's own resolved text.
#[test]
fn a_migrated_store_reads_as_the_fresh_one() {
    let fx = Fixture::new("dev_migrate");
    fx.write("fixed.toml", "gamma = 0.2\nrho = 0.5\n");
    fx.write("fit.toml", &fit_config("from_file = \"fixed.toml\""));
    let leaves = fx.fit(&["fit.toml", "--seed", "1", "--sweep", "rho=0.4,0.6"]);
    assert_eq!(leaves.len(), 2, "{leaves:?}");
    let fresh: Vec<_> = leaves.iter().map(|l| fx.predict(l).unwrap()).collect();
    assert!(fresh.iter().all(|p| !p.is_empty()) && fresh[0] != fresh[1], "two points, two predictions");

    let results = fx.dir.join("results");
    let written = to_layout_zero(&results);
    assert_eq!(written.len(), 2, "one fit folder per sweep point");

    let err = fx.predict(&leaves[0]).expect_err("a layout-0 folder");
    let folder = fx.folder(&leaves[0]);
    assert!(err.contains("store layout 0") && err.contains(&format!("camdl dev migrate {}", folder.display())),
        "{err}");

    let before = files_under(&results);
    let dry = fx.migrate(&["results", "--dry-run"]);
    let said = text(&dry);
    assert!(dry.status.success(), "{said}");
    assert!(said.contains("would migrate") && said.contains("writing fit.resolved.toml"), "{said}");
    assert!(said.contains("dry run: nothing written"), "{said}");
    assert_eq!(files_under(&results), before, "--dry-run writes nothing");

    let run = fx.migrate(&["results"]);
    let said = text(&run);
    assert!(run.status.success(), "{said}");
    assert!(said.contains("0 refused"), "{said}");
    for (folder, fit_written) in &written {
        let migrated = std::fs::read_to_string(folder.join("fit.resolved.toml")).unwrap();
        assert!(migrated.contains("Written by `camdl dev migrate` (store layout 0 to 1)"), "{migrated}");
        assert!(migrated.contains("fit.toml.original") && migrated.contains("fit.meta.json"), "{migrated}");
        assert_eq!(body(&migrated), body(fit_written), "{}: the fit's own text", folder.display());
        assert_eq!(json(&folder.join("fit.meta.json"))["store_layout"], 1);
    }
    for record in named(&results, "run.json") {
        assert_eq!(json(&record)["store_layout"], 1, "{}", record.display());
    }
    for (leaf, fresh) in leaves.iter().zip(&fresh) {
        assert_eq!(&fx.predict(leaf).unwrap(), fresh, "{leaf}: the migrated folder reads as the fresh one");
    }

    let again = fx.migrate(&["results"]);
    let said = text(&again);
    assert!(again.status.success() && said.contains("0 folder(s) migrated"), "{said}");
}

/// A folder whose fit file is not the one its record names is refused and
/// left at layout 0; the folder beside it is migrated; the command exits 1.
#[test]
fn a_folder_whose_records_disagree_is_refused_and_the_rest_are_migrated() {
    let fx = Fixture::new("dev_migrate");
    fx.write("a.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    fx.write("b.toml", &fit_config("gamma = 0.25\nrho = 0.5"));
    let a = fx.fit(&["a.toml", "--seed", "1"]).remove(0);
    let b = fx.fit(&["b.toml", "--seed", "1"]).remove(0);
    let results = fx.dir.join("results");
    to_layout_zero(&results);
    let (fa, fb) = (fx.folder(&a), fx.folder(&b));
    let original = fa.join("fit.toml.original");
    let edited = format!("{}# edited after the fit\n", std::fs::read_to_string(&original).unwrap());
    std::fs::write(&original, edited).unwrap();

    let run = fx.migrate(&["results"]);
    let said = text(&run);
    assert_eq!(run.status.code(), Some(1), "{said}");
    assert!(said.contains(&format!("refused {}", fa.strip_prefix(&fx.dir).unwrap().display())) && said.contains("fit_toml_hash"), "{said}");
    assert!(!fa.join("fit.resolved.toml").exists(), "a refused folder is left as it was");
    assert!(json(&fa.join("fit.meta.json")).get("store_layout").is_none());
    assert!(fb.join("fit.resolved.toml").is_file(), "the other folder is migrated: {said}");
    assert_eq!(json(&fb.join("fit.meta.json"))["store_layout"], 1);
    fx.predict(&b).unwrap();
}

/// A folder at a higher layout than this camdl's was written by a newer one:
/// the reader and the migration refuse it, and it is left as it was.
#[test]
fn a_folder_from_a_newer_camdl_is_refused() {
    let fx = Fixture::new("dev_migrate");
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let folder = fx.folder(&leaf);
    let meta = folder.join("fit.meta.json");
    let mut v = json(&meta);
    v["store_layout"] = serde_json::json!(2);
    set_json(&meta, &v);
    let before = std::fs::read(&meta).unwrap();

    let err = fx.predict(&leaf).expect_err("a newer layout");
    assert!(err.contains("store layout 2") && err.contains("written by a newer camdl"), "{err}");
    let run = fx.migrate(&[&folder.to_string_lossy()]);
    let said = text(&run);
    assert_eq!(run.status.code(), Some(1), "{said}");
    assert!(said.contains("written by a newer camdl"), "{said}");
    assert_eq!(std::fs::read(&meta).unwrap(), before);
}

/// A layout-0 folder that already holds a `fit.resolved.toml` (one the fit
/// wrote, or one a migration wrote and then stopped before raising the
/// layout) keeps it when it agrees with the folder's records, and is refused
/// when it does not.
#[test]
fn a_resolved_config_already_in_a_layout_zero_folder_is_checked() {
    let fx = Fixture::new("dev_migrate");
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1", "--allow-forcing-extrapolation"]).remove(0);
    let folder = fx.folder(&leaf);
    let results = fx.dir.join("results");
    // A branch build's folder: the resolved file, the flag folded into it, and
    // no layout number.
    let fit_written = std::fs::read_to_string(folder.join("fit.resolved.toml")).unwrap();
    let fit_written = &fit_written;
    assert!(fit_written.contains("allow_forcing_extrapolation = true"), "{fit_written}");
    without_layout_numbers(&results);

    std::fs::write(folder.join("fit.resolved.toml"), fit_written.replace("rho = 0.5", "rho = 0.7")).unwrap();
    let run = fx.migrate(&["results"]);
    let said = text(&run);
    assert_eq!(run.status.code(), Some(1), "{said}");
    assert!(said.contains("already exists and says something other than its records"), "{said}");

    std::fs::write(folder.join("fit.resolved.toml"), fit_written).unwrap();
    let run = fx.migrate(&["results"]);
    assert!(run.status.success(), "{}", text(&run));
    assert_eq!(&std::fs::read_to_string(folder.join("fit.resolved.toml")).unwrap(), fit_written,
        "an agreeing file is kept as it was written");
    fx.predict(&leaf).unwrap();
}

/// A folder main wrote under `--allow-forcing-extrapolation` stored its fit
/// file with the flag folded in, while `fit_toml_hash` hashes the file as
/// read: the file the fit read is in no record. The migration refuses it,
/// saying so rather than that the file was edited, and leaves it as it was.
#[test]
fn a_folder_main_wrote_with_the_flag_folded_in_is_refused_as_such() {
    let fx = Fixture::new("dev_migrate");
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1", "--allow-forcing-extrapolation"]).remove(0);
    let folder = fx.folder(&leaf);
    let results = fx.dir.join("results");
    as_main_folded_it(&results);
    let original = std::fs::read(folder.join("fit.toml.original")).unwrap();

    let run = fx.migrate(&["results"]);
    let said = text(&run);
    assert_eq!(run.status.code(), Some(1), "{said}");
    assert!(said.contains("folded --allow-forcing-extrapolation into the fit file it stored"), "{said}");
    let rel = folder.strip_prefix(&fx.dir).unwrap().display().to_string();
    let line = said.lines().find(|l| l.starts_with("refused ")).expect("a refused line");
    assert_eq!(line.matches(rel.as_str()).count(), 1, "the folder is named once: {line}");
    assert!(said.contains("Re-run the fit"), "{said}");
    assert!(!said.contains("is not the file the fit read"), "not a tampering refusal: {said}");
    assert!(!folder.join("fit.resolved.toml").exists());
    assert_eq!(std::fs::read(folder.join("fit.toml.original")).unwrap(), original);
    assert!(json(&folder.join("fit.meta.json")).get("store_layout").is_none());
}
