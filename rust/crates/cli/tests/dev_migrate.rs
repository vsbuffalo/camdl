//! `camdl dev migrate`, end to end through the release binary.
//!
//! A store at layout 0 is made the only way one can be made now: by a fit run
//! of this build, then rewritten to what an older camdl left
//! ([`to_layout_zero`]: no `store_layout` field in any record, and no
//! `fit.resolved.toml` in any fit folder). That is the whole difference
//! between layouts 0 and 1, so the function is the fixture's generator, and
//! it is re-run by every test rather than committed as files.
//!
//! What is pinned:
//!
//! 1. A reader of a stored fit's config refuses a layout-0 folder naming
//!    `camdl dev migrate <folder>`; after the migration the reader gives the
//!    output it gave the fresh folder, and the migrated `fit.resolved.toml`
//!    is the fit's own text under a header that says a migration wrote it,
//!    for a sweep point, a `from_file` `[fixed]` and a command-line problem
//!    flag alike. A second run changes nothing.
//! 2. `--dry-run` lists the folders and writes nothing.
//! 3. A folder whose records disagree is refused, left as it was, and the
//!    command exits 1, while the other folders are migrated.
//! 4. A folder written by a newer camdl (a higher layout) is refused, by the
//!    reader and by the migration.
//! 5. A `fit.resolved.toml` already in a layout-0 folder (a fit-written one,
//!    or one a migration wrote before it stopped) is kept when it agrees with
//!    the records, and refuses the folder when it does not.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability
}

let N = S + I + R

transitions {
  infection : S --> I @ beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10 }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }
}

simulate { from = 0 'days  to = 70 'days  dt = 1 'days }
"#;

const METHOD: &str = "[method]\nalgorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nparticles = 20\n\
                      chains = 1\niterations = 20\nburn_in = 5\n";

/// A fit config: `fixed` is the body of its `[fixed]`.
fn fit_config(fixed: &str) -> String {
    format!(
        "[model]\ncamdl = \"m.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.3, prior = {{ uniform = {{}} }} }}\n[fixed]\n{fixed}\n{METHOD}"
    )
}

struct Fixture {
    dir: PathBuf,
    camdl: PathBuf,
    camdlc: PathBuf,
}

impl Fixture {
    fn new() -> Fixture {
        static N: AtomicUsize = AtomicUsize::new(0);
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        let version = Command::new(&camdl).arg("--version").output().unwrap();
        let version = String::from_utf8_lossy(&version.stdout);
        assert!(
            version.contains(&format!("+{} ", env!("CAMDL_SOURCE_LABEL"))),
            "{} is not built from this source ({}): {version} — run `make build`",
            camdl.display(),
            env!("CAMDL_SOURCE_LABEL")
        );
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("dev_migrate-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("home")).unwrap();
        let fx = Fixture { dir, camdl, camdlc };
        fx.write("m.camdl", MODEL);
        let cases: String = (1..=10).map(|w| format!("{}\t{}\n", 7 * w, 2 + 3 * w)).collect();
        fx.write("cases.tsv", &format!("time\tcases\n{cases}"));
        fx
    }

    fn write(&self, name: &str, text: &str) {
        std::fs::write(self.dir.join(name), text).unwrap();
    }

    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(&self.camdl)
            .current_dir(&self.dir)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", self.dir.join("home"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &self.camdlc)
            .env("CAMDL_IR_CACHE_DIR", self.dir.join("ir-cache"))
            .args(args)
            .output()
            .expect("spawn camdl")
    }

    /// `fit run`, returning every leaf it stored or reused, relative to the
    /// fixture directory.
    fn fit(&self, args: &[&str]) -> Vec<String> {
        let mut all = vec!["fit", "run"];
        all.extend(args);
        all.extend(["--progress", "none"]);
        let stderr = ok(&self.camdl(&all));
        let leaves: Vec<String> = stderr
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.strip_prefix("stored ").and_then(|r| r.split(" · ").nth(1)).map(str::to_string)
                    .or_else(|| l.strip_prefix("cache hit — reusing ").map(|r| format!("results/{r}")))
            })
            .collect();
        assert!(!leaves.is_empty(), "no stored leaf in:\n{stderr}");
        leaves
    }

    /// The fit folder holding `leaf`.
    fn folder(&self, leaf: &str) -> PathBuf {
        self.dir.join(leaf).ancestors().nth(2).unwrap().to_path_buf()
    }

    /// `fit predict <leaf>`'s files, by path under the folder's `predict/`.
    fn predict(&self, leaf: &str) -> Result<BTreeMap<String, Vec<u8>>, String> {
        let folder = self.folder(leaf);
        let _ = std::fs::remove_dir_all(folder.join("predict"));
        let out = self.camdl(&["fit", "predict", &self.dir.join(leaf).to_string_lossy(), "--n-draws", "5",
                               "--seed", "1", "--progress", "none"]);
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).into_owned());
        }
        Ok(files_under(&folder.join("predict")))
    }

    fn migrate(&self, args: &[&str]) -> Output {
        let mut all = vec!["dev", "migrate"];
        all.extend(args);
        self.camdl(&all)
    }
}

fn ok(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "camdl failed:\n{stderr}\n{}", String::from_utf8_lossy(&out.stdout));
    stderr
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn files_under(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.strip_prefix(dir).unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// Every file named `name` under `dir`.
fn named(dir: &Path, name: &str) -> Vec<PathBuf> {
    files_under(dir).into_keys().filter(|k| k.ends_with(name)).map(|k| dir.join(k)).collect()
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn set_json(path: &Path, value: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

/// Rewrite the store at `results` to what a camdl before store layouts left:
/// no `store_layout` in any `run.json` or `fit.meta.json`, and no
/// `fit.resolved.toml`. Returns each fit folder's `fit.resolved.toml` as the
/// fit wrote it.
fn to_layout_zero(results: &Path) -> BTreeMap<PathBuf, String> {
    let mut resolved = BTreeMap::new();
    for record in named(results, "run.json").into_iter().chain(named(results, "fit.meta.json")) {
        let mut v = json(&record);
        assert_eq!(v["store_layout"], 1, "{} is written at layout 1", record.display());
        v.as_object_mut().unwrap().remove("store_layout");
        set_json(&record, &v);
    }
    for file in named(results, "fit.resolved.toml") {
        resolved.insert(file.parent().unwrap().to_path_buf(), std::fs::read_to_string(&file).unwrap());
        std::fs::remove_file(&file).unwrap();
    }
    resolved
}

/// The text below a `fit.resolved.toml`'s opening comment.
fn body(text: &str) -> String {
    text.lines().skip_while(|l| l.starts_with('#')).collect::<Vec<_>>().join("\n")
}

/// A sweep over a `from_file` parameter, under a command-line problem flag,
/// migrated from layout 0: the readers refuse it naming the migration, a dry
/// run writes nothing, and after the migration `fit predict` gives what it
/// gave the fresh folders, from the fit's own resolved text.
#[test]
fn a_migrated_store_reads_as_the_fresh_one() {
    let fx = Fixture::new();
    fx.write("fixed.toml", "gamma = 0.2\nrho = 0.5\n");
    fx.write("fit.toml", &fit_config("from_file = \"fixed.toml\""));
    let leaves = fx.fit(&["fit.toml", "--seed", "1", "--sweep", "rho=0.4,0.6", "--allow-forcing-extrapolation"]);
    assert_eq!(leaves.len(), 2, "{leaves:?}");
    let fresh: Vec<_> = leaves.iter().map(|l| fx.predict(l).unwrap()).collect();
    assert!(fresh.iter().all(|p| !p.is_empty()) && fresh[0] != fresh[1], "two points, two predictions");

    let results = fx.dir.join("results");
    let written = to_layout_zero(&results);
    assert_eq!(written.len(), 2, "one fit folder per sweep point");
    for (folder, text) in &written {
        assert!(text.contains("allow_forcing_extrapolation = true"), "{}: {text}", folder.display());
    }

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
    let fx = Fixture::new();
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
    let fx = Fixture::new();
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
    let fx = Fixture::new();
    fx.write("fit.toml", &fit_config("gamma = 0.2\nrho = 0.5"));
    let leaf = fx.fit(&["fit.toml", "--seed", "1"]).remove(0);
    let folder = fx.folder(&leaf);
    let results = fx.dir.join("results");
    let written = to_layout_zero(&results);
    let fit_written = &written[&folder];

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
