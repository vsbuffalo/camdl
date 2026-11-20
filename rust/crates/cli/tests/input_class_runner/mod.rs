//! The differential runner the `pfilter`, `profile` and `survey` input-class
//! suites share (`pfilter_input_classes.rs`, `profile_input_classes.rs`,
//! `survey_input_classes.rs`; proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests").
//!
//! [`run`] runs the release `camdl` with its working directory a fresh
//! temporary store, in a cleared environment holding an explicit allowlist, a
//! private `HOME` and compiled-IR cache, and the compiler under test, and
//! returns every stored leaf: its kind, its run id, its level hashes and the
//! bytes its manifest checksums (each file's SHA-256, checked against the
//! digest `run.json` records). The assertions, one per class, compare the
//! leaves of one kind — the command's own (`pfilter`, `profile_point`,
//! `survey`) — between two invocations.
//!
//! The release binary under test must be built from the source this test was
//! built from: [`Fixture::create`] checks that `--version` names the source
//! fingerprint both builds embed.

#![allow(dead_code)] // each suite uses the part it needs

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// A suite's fixture files, written under the target directory's scratch
/// space (`CARGO_TARGET_TMPDIR`) and removed when the last test holding them
/// is done.
pub struct Fixture {
    pub dir: PathBuf,
    pub camdl: PathBuf,
    pub camdlc: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Fixture {
    /// A fresh directory named after the suite, the files `files` gives for
    /// it written into it, and each `.camdl` in `compile` compiled beside it
    /// to `<stem>.ir.json`.
    pub fn create(suite: &str, files: impl FnOnce(&Path) -> Vec<(&'static str, String)>, compile: &[&str]) -> Fixture {
        static N: AtomicUsize = AtomicUsize::new(0);
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        // The binary under test is built from the source this test was built
        // from: `--version` names the source fingerprint both builds embed.
        let version = Command::new(&camdl).arg("--version").output().unwrap();
        let version = String::from_utf8_lossy(&version.stdout);
        assert!(
            version.contains(&format!("+{} ", env!("CAMDL_SOURCE_LABEL"))),
            "{} is not built from this source ({}): {version} — run `make build`",
            camdl.display(),
            env!("CAMDL_SOURCE_LABEL")
        );
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "{suite}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in files(&dir) {
            std::fs::write(dir.join(name), text).unwrap();
        }
        for name in compile {
            let src = dir.join(name);
            let out = Command::new(&camdlc).env_clear().arg(&src).output().unwrap();
            assert!(out.status.success(), "camdlc {name}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(src.with_extension("ir.json"), out.stdout).unwrap();
        }
        Fixture { dir, camdl, camdlc }
    }

    /// `name` in the fixture directory.
    pub fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }
}

/// One invocation: the command and its arguments, and the variables it sets
/// in the child's otherwise cleared environment.
#[derive(Clone)]
pub struct Invocation {
    pub fx: Arc<Fixture>,
    pub args: Vec<String>,
    pub set: Vec<(String, String)>,
    /// Allowlisted variables this invocation drops.
    pub drop: Vec<String>,
}

impl Invocation {
    pub fn new(fx: &Arc<Fixture>, args: &[&str]) -> Invocation {
        Invocation { fx: fx.clone(), args: args.iter().map(|s| s.to_string()).collect(), set: Vec::new(), drop: Vec::new() }
    }
    /// The same invocation with `extra` appended.
    pub fn with(&self, extra: &[&str]) -> Invocation {
        let mut inv = self.clone();
        inv.args.extend(extra.iter().map(|s| s.to_string()));
        inv
    }
    /// The same invocation with the value after `flag` replaced by `value`.
    pub fn set(&self, flag: &str, value: &str) -> Invocation {
        let mut inv = self.clone();
        let at = inv.args.iter().position(|a| a == flag).unwrap_or_else(|| panic!("{flag} in {:?}", inv.args)) + 1;
        inv.args[at] = value.to_string();
        inv
    }
    pub fn env(mut self, k: &str, v: &str) -> Self {
        self.set.push((k.into(), v.into()));
        self
    }
    pub fn without(mut self, k: &str) -> Self {
        self.drop.push(k.into());
        self
    }
}

/// The variables a run may inherit from the test process: what starting a
/// process needs, and nothing that configures camdl.
const ALLOWED: &[&str] = &["PATH", "TMPDIR"];

/// A `camdl` command in `cwd`, in a cleared environment holding [`ALLOWED`],
/// a private `HOME` and compiled-IR cache, and the compiler under test.
pub fn command(fx: &Fixture, cwd: &Path, set: &[(String, String)], drop: &[String]) -> Command {
    let mut cmd = Command::new(&fx.camdl);
    cmd.current_dir(cwd).env_clear();
    for k in ALLOWED.iter().filter(|k| !drop.iter().any(|d| d == *k)) {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    cmd.env("HOME", fx.dir.join("home"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &fx.camdlc)
        // A private compiled-IR cache: never the user's, never shared with
        // another test process.
        .env("CAMDL_IR_CACHE_DIR", fx.dir.join("ir-cache"));
    for k in drop {
        cmd.env_remove(k);
    }
    for (k, v) in set {
        cmd.env(k, v);
    }
    cmd
}

/// Run `inv` in the store `store` (its working directory, under which
/// `results/` is the default store root) and return the output.
pub fn run_in(store: &Path, inv: &Invocation) -> Output {
    let mut cmd = command(&inv.fx, store, &inv.set, &inv.drop);
    cmd.args(&inv.args);
    if !inv.args.iter().any(|a| a == "--progress" || a == "--no-progress") {
        cmd.args(["--progress", "none"]);
    }
    cmd.output().expect("spawn camdl")
}

/// [`run_in`], requiring success, and every leaf the store holds afterwards.
pub fn run_into(store: &Path, inv: &Invocation) -> Stored {
    let out = run_in(store, inv);
    assert!(
        out.status.success(),
        "camdl {:?} (env {:?}) failed:\n{}",
        inv.args,
        inv.set,
        String::from_utf8_lossy(&out.stderr)
    );
    Stored { leaves: leaves(store), stderr: String::from_utf8_lossy(&out.stderr).into_owned() }
}

/// [`run_into`] a fresh store.
pub fn run(inv: &Invocation) -> Stored {
    let store = tempfile::tempdir().unwrap();
    run_into(store.path(), inv)
}

/// A stored leaf.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub kind: String,
    pub run_id: String,
    /// Each level's name and hash, in path order.
    pub levels: BTreeMap<String, String>,
    /// Each file the manifest checksums, with the SHA-256 of its bytes.
    pub files: BTreeMap<String, String>,
    /// The leaf's `run.json`, parsed.
    pub record: serde_json::Value,
    pub dir: PathBuf,
}

#[derive(Debug)]
pub struct Stored {
    pub leaves: Vec<Leaf>,
    pub stderr: String,
}

impl Stored {
    /// The leaves of `kind`, by run id.
    pub fn of(&self, kind: &str) -> BTreeMap<String, &Leaf> {
        self.leaves.iter().filter(|l| l.kind == kind).map(|l| (l.run_id.clone(), l)).collect()
    }
    pub fn ids(&self, kind: &str) -> BTreeSet<String> {
        self.of(kind).into_keys().collect()
    }
    /// The hashes the leaves of `kind` hold at `level`.
    pub fn level_hashes(&self, kind: &str, level: &str) -> BTreeSet<String> {
        self.of(kind)
            .values()
            .map(|l| l.levels.get(level).unwrap_or_else(|| panic!("{kind} leaf has no `{level}` level: {:?}", l.levels)).clone())
            .collect()
    }
    /// The stored bytes of every leaf of `kind`, sorted, without their run
    /// ids: what two runs computed, whatever each was keyed by.
    pub fn bytes(&self, kind: &str) -> Vec<BTreeMap<String, String>> {
        let mut out: Vec<BTreeMap<String, String>> = self.of(kind).values().map(|l| l.files.clone()).collect();
        out.sort();
        out
    }
    /// The kind, run id and stored bytes of every leaf of `kind`.
    pub fn result(&self, kind: &str) -> Vec<(String, BTreeMap<String, String>)> {
        self.of(kind).values().map(|l| (l.run_id.clone(), l.files.clone())).collect()
    }
}

/// Every leaf under `root`: a directory holding a `run.json`, outside a
/// quarantined or earlier attempt.
pub fn leaves(root: &Path) -> Vec<Leaf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() || p.file_name().is_some_and(|n| n == ".quarantine" || n == ".staging" || n == ".attempts") {
                continue;
            }
            let record = p.join("run.json");
            if record.is_file() {
                if let Some(l) = leaf(&p, &record) {
                    out.push(l);
                }
            }
            stack.push(p);
        }
    }
    out.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    out
}

/// The leaf recorded at `record`, or `None` for a record that is not a leaf's
/// (a profile invocation's).
fn leaf(dir: &Path, record: &Path) -> Option<Leaf> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(record).unwrap()).unwrap();
    let (Some(kind), Some(run_id), Some(levels)) = (v["kind"].as_str(), v["run_id"].as_str(), v["levels"].as_array()) else {
        return None;
    };
    let mut files = BTreeMap::new();
    for (name, entry) in v["artifacts"].as_object().into_iter().flatten() {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let digest = hex(&Sha256::digest(&bytes));
        assert_eq!(Some(digest.as_str()), entry["digest"].as_str(), "{}: {name} is the manifest's bytes", dir.display());
        files.insert(name.clone(), digest);
    }
    let levels = levels
        .iter()
        .map(|l| (l["name"].as_str().unwrap().to_string(), l["hash"].as_str().unwrap().to_string()))
        .collect();
    Some(Leaf { kind: kind.to_string(), run_id: run_id.to_string(), levels, files, record: v.clone(), dir: dir.to_path_buf() })
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The class of every input of `command` (its subcommand path, `["fit",
/// "run"]`) and of every environment variable (`env:<field>`), in the spelling
/// of `#[input(..)]`, as the release `camdl` prints them with its hidden
/// `__input-classes` mode from the inventory `#[derive(Inputs)]` generates
/// (`src/input_class.rs`). An integration test cannot import the crate, so it
/// asks the binary it tests, which [`Fixture::create`] checks is built from
/// this source.
pub fn declared_classes(command: &[&str]) -> BTreeMap<String, String> {
    let camdl = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    let out = Command::new(&camdl).arg("__input-classes").args(command).output().unwrap();
    assert!(
        out.status.success(),
        "camdl __input-classes {}: {}",
        command.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| {
            let (id, class) = l.split_once('\t').unwrap_or_else(|| panic!("not `<id>\\t<class>`: {l}"));
            (id.to_string(), class.to_string())
        })
        .collect()
}

/// The level the inventory declares for a keyed input: `input` is an argument
/// id of `command`, or `env:<field>` for an environment variable. An
/// environment variable keyed at a simulation's `model` level enters the
/// command at `env_model_level`.
pub fn declared_level(command: &[&str], input: &str, env_model_level: &str) -> String {
    let classes = declared_classes(command);
    let class = classes
        .get(input)
        .unwrap_or_else(|| panic!("`{input}` is not an input of `camdl {}`", command.join(" ")));
    let level = class
        .strip_prefix("keyed(")
        .and_then(|l| l.strip_suffix(')'))
        .unwrap_or_else(|| panic!("`{input}` is classified {class}, not keyed"));
    match level {
        "model" if input.starts_with("env:") => env_model_level.to_string(),
        _ => level.to_string(),
    }
}

// ─── The assertions, one per class ──────────────────────────────────────────

/// `keyed(level)`: every `kind` leaf of `b` has a run id `a` does not, and a
/// hash at `level` that `a` does not.
pub fn assert_rekeyed(kind: &str, input: &str, level: &str, a: &Stored, b: &Stored) {
    let (ia, ib) = (a.ids(kind), b.ids(kind));
    assert!(!ia.is_empty() && !ib.is_empty(), "both runs store a {kind} leaf");
    assert!(ia.is_disjoint(&ib), "{input} is keyed, so it re-keys every leaf: {ia:?} vs {ib:?}");
    let (la, lb) = (a.level_hashes(kind, level), b.level_hashes(kind, level));
    assert!(la.is_disjoint(&lb), "{input} is classified keyed({level}), so it re-keys the `{level}` level: {la:?} vs {lb:?}");
}

/// `execution`: the same run ids and the same stored bytes.
pub fn assert_same_result(kind: &str, a: &Stored, b: &Stored) {
    assert!(!a.of(kind).is_empty(), "the run stores a {kind} leaf");
    assert_eq!(a.result(kind), b.result(kind), "an execution input leaves the run ids and the stored bytes unchanged");
}

/// `presentation` / `provenance`: the same leaves, run ids and stored bytes.
pub fn assert_leaf_unaffected(kind: &str, input: &str, a: &Stored, b: &Stored) {
    assert!(!a.of(kind).is_empty(), "the run stores a {kind} leaf");
    assert_eq!(a.result(kind), b.result(kind), "{input} leaves the run ids and the stored bytes unchanged");
}

/// `expands`: every `kind` leaf of the smaller set is one of the larger, under
/// the same run id, and the larger set has more.
pub fn assert_shared_leaves_keep_ids(kind: &str, small: &Stored, large: &Stored) {
    let (a, b) = (small.ids(kind), large.ids(kind));
    assert!(!a.is_empty(), "the smaller set stores a leaf");
    assert!(a.is_subset(&b), "the shared leaves keep their run ids: {a:?} not within {b:?}");
    assert!(b.len() > a.len(), "the larger set has more leaves");
}

/// `removed`: the run is refused, the message names the old spelling, and
/// the store holds nothing.
pub fn assert_removed(spelling: &str, with: &Invocation) {
    let store = tempfile::tempdir().unwrap();
    let out = run_in(store.path(), with);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{spelling} is refused:\n{err}");
    assert!(err.contains(spelling), "the refusal names {spelling}:\n{err}");
    assert!(leaves(store.path()).is_empty(), "a refused run stores nothing");
}

/// The value the fit folder holding the fit leaf `leaf` records for the fixed
/// parameter `name` (`fit.meta.json` `fixed`): a sweep point's value.
pub fn segment_fixed(leaf: &Path, name: &str) -> Option<f64> {
    let segment = leaf.parent()?.parent()?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(segment.join("fit.meta.json")).ok()?).ok()?;
    meta["fixed"][name].as_f64()
}

/// The fit leaves of a `fit run --sweep NAME=a,b` in `stored`, as the leaf at
/// the file's own value `archived` and the leaf at the other point.
pub fn sweep_points(stored: &Stored, name: &str, archived: f64) -> (Leaf, Leaf) {
    let fits: Vec<Leaf> = stored.of("fit_method").into_values().cloned().collect();
    assert_eq!(fits.len(), 2, "one fit leaf per sweep point");
    let (same, other): (Vec<Leaf>, Vec<Leaf>) =
        fits.into_iter().partition(|l| segment_fixed(&l.dir, name) == Some(archived));
    assert_eq!((same.len(), other.len()), (1, 1), "one point at the archived value, one elsewhere");
    (same.into_iter().next().unwrap(), other.into_iter().next().unwrap())
}

/// The entries of a private IR cache that hold compiled IR.
pub fn cached_ir(cache: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(cache)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json") && !p.to_string_lossy().ends_with(".deps.json"))
        .collect()
}
