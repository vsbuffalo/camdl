//! A leaf is found by its hashes, not by its labels (proposal
//! `docs/dev/proposals/2026-10-24-batch-to-simulate.md` §3 "Arm labels",
//! §6 test 11).
//!
//! A leaf's `run_id` is its five level hashes alone; its directory is
//! `{label}-{hash8}` per level, and the label is provenance. Before
//! `runid::find_leaf`, both reuse checks and the write seam built the path
//! from the *current* labels, so a leaf whose label had changed while its
//! content had not — a renamed model file (the model level's label is the
//! file stem), or an arm relabelled from `baseline` — was a miss although the
//! store held it, and a re-run stored the same identity a second time under
//! the new name.
//!
//! The proposal also lists a renamed scenario preset among the label-only
//! changes. It is not one: a preset's `name` enters the whole-IR model digest
//! (`runid::ir_hash`, `impl ContentAddressed for Preset` writes `name`), so
//! renaming a preset re-keys the model level and every leaf of that model is
//! a genuine miss. The relabelled-arm case below exercises what that case
//! was meant to: a scenario directory whose label changed under a fixed
//! identity.
//!
//! Shells out to the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        bin.display()
    );
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta  : rate in [0.01, 5.0]
  gamma : rate in [0.01, 5.0]
}
init { S = 990  I = 10  R = 0 }
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
simulate { from = 0 'days  to = 20 'days }
"#;

const PARAMS: &str = "beta = 0.4\ngamma = 0.2\n";

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new(model_stem: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(format!("{model_stem}.camdl")), MODEL).unwrap();
        std::fs::write(dir.path().join("p.toml"), PARAMS).unwrap();
        Fixture { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// `simulate <model> --params p.toml --replicates 2 --seed 7 --output-dir out <extra…>`.
    fn simulate(&self, model: &str, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate", model, "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "1", "--seed", "7", "--replicates", "2", "--output-dir", "out",
        ];
        args.extend_from_slice(extra);
        Command::new(skip_if_missing_binary())
            .args(&args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    /// Every `Sim` leaf under the output root: its directory and its `run_id`,
    /// sorted by `run_id`.
    fn sim_leaves(&self) -> Vec<(PathBuf, String)> {
        let mut out: Vec<(PathBuf, String)> = run_leaves(&self.path("out").join("sims"))
            .into_iter()
            .map(|dir| {
                let rec: serde_json::Value = serde_json::from_str(
                    &std::fs::read_to_string(dir.join("run.json")).unwrap(),
                )
                .unwrap();
                (dir, rec["run_id"].as_str().unwrap().to_string())
            })
            .collect();
        out.sort_by(|a, b| a.1.cmp(&b.1));
        out
    }
}

fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    out
}

fn assert_ok(out: &Output, what: &str) {
    assert!(
        out.status.success(),
        "{what}: exited {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The model level's label is the model file's stem. Renaming the file changes
/// the label and nothing else, so a second run over the renamed file finds
/// every cell the first run stored: the store holds the same two leaves, under
/// their original directories, and nothing is stored a second time under the
/// new name.
#[test]
fn renaming_the_model_file_keeps_every_leaf_a_hit() {
    let fx = Fixture::new("sir");
    assert_ok(&fx.simulate("sir.camdl", &[]), "first run");
    let before = fx.sim_leaves();
    assert_eq!(before.len(), 2, "two replicates are two leaves");
    assert!(
        before.iter().all(|(dir, _)| dir.to_string_lossy().contains("/sims/sir-")),
        "the model level is labelled by the file stem: {before:?}"
    );

    std::fs::rename(fx.path("sir.camdl"), fx.path("sir_renamed.camdl")).unwrap();
    assert_ok(&fx.simulate("sir_renamed.camdl", &[]), "second run over the renamed file");

    let after = fx.sim_leaves();
    assert_eq!(
        after, before,
        "the renamed file's run must land on the leaves the store already holds — \
         same directories, same run_ids — not a second copy under `sir_renamed-…`"
    );
    assert!(
        !fx.path("out").join("sims").read_dir().unwrap().flatten().any(|e| {
            e.file_name().to_string_lossy().starts_with("sir_renamed-")
        }),
        "no `sir_renamed-…` model directory may appear"
    );
}

/// The scenario level's label is provenance too. A leaf whose scenario
/// directory was relabelled on disk — the label in the segment and in
/// `run.json` both changed, the hashes untouched — is found by the next run
/// and not stored again under the current label.
#[test]
fn a_relabelled_scenario_directory_is_still_a_hit() {
    let fx = Fixture::new("sir");
    assert_ok(&fx.simulate("sir.camdl", &[]), "first run");
    let before = fx.sim_leaves();
    assert_eq!(before.len(), 2);

    // Relabel: the leaves' parent is the scenario directory `_none_-<h8>`
    // (the implicit arm `(none)`, as the path sanitizer renders it).
    let scenario_dir = before[0].0.parent().unwrap().to_path_buf();
    let seg = scenario_dir.file_name().unwrap().to_string_lossy().into_owned();
    let (label, h8) = seg.rsplit_once('-').unwrap();
    assert_eq!(label, "_none_");
    let relabelled = scenario_dir.with_file_name(format!("renamed_arm-{h8}"));
    std::fs::rename(&scenario_dir, &relabelled).unwrap();
    for leaf in run_leaves(&relabelled) {
        let path = leaf.join("run.json");
        let mut rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for level in rec["levels"].as_array_mut().unwrap() {
            if level["name"] == "scenario" {
                level["label"] = serde_json::Value::String("renamed_arm".into());
            }
        }
        std::fs::write(&path, serde_json::to_string_pretty(&rec).unwrap()).unwrap();
    }
    let relabelled_leaves = fx.sim_leaves();
    assert_eq!(relabelled_leaves.len(), 2);
    assert!(relabelled_leaves.iter().all(|(d, _)| d.starts_with(&relabelled)));

    assert_ok(&fx.simulate("sir.camdl", &[]), "second run against the relabelled store");
    let after = fx.sim_leaves();
    assert_eq!(
        after, relabelled_leaves,
        "the run must find the relabelled leaves by hash and store nothing new; \
         a `_none_-<h8>` directory must not reappear beside `renamed_arm-<h8>`"
    );
    assert!(!scenario_dir.exists(), "no second copy under the current label");
}

/// A store that renamed nothing sees exactly the hits it always did: a second
/// identical run changes nothing on disk.
#[test]
fn an_unrenamed_store_has_the_same_hits_as_before() {
    let fx = Fixture::new("sir");
    assert_ok(&fx.simulate("sir.camdl", &[]), "first run");
    let before = fx.sim_leaves();
    assert_ok(&fx.simulate("sir.camdl", &[]), "second identical run");
    assert_eq!(fx.sim_leaves(), before, "an identical re-run adds and moves nothing");
}
