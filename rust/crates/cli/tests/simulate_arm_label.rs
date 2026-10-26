//! The derived label of `simulate`'s ad-hoc arm (proposal
//! `docs/dev/proposals/2026-10-24-batch-to-simulate.md` §3 "Arm labels";
//! §6 tests 3 and 10).
//!
//! Without `--scenario` the run's one arm is labelled from what
//! `--enable`/`--disable`/`--param` built: `(none)` for none of them,
//! `(none)+vaccination`, `(none);beta=0.3`. Two different ad-hoc arms are
//! told apart in the leaf path, `run.json` and the listings, and the
//! parenthesised label cannot collide with a declared scenario (whose name is
//! an identifier). It is provenance: the arm's identity is its resolved
//! delta, so a preset-named arm with the same delta is the same leaf, and a
//! leaf written under an earlier label (`baseline`) is found by hash and
//! reused.
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

/// SIR + V with a vaccination campaign (inactive by default) and a preset
/// `vax` that enables it — the same delta `--enable vaccination` builds.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta  : rate in [0.01, 5.0]
  gamma : rate in [0.01, 5.0]
}

init { S = 990  I = 10  R = 0  V = 0 }

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 30 'days }

interventions {
  vaccination : transfer(fraction = 0.5, from = S, to = V) at [15]
}

scenarios {
  vax { enable = [vaccination] }
}
"#;

const PARAMS: &str = "beta = 0.4\ngamma = 0.2\n";

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m.camdl"), MODEL).unwrap();
        std::fs::write(dir.path().join("p.toml"), PARAMS).unwrap();
        Fixture { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn camdl(&self, args: &[&str]) -> Output {
        Command::new(skip_if_missing_binary())
            .args(args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    fn simulate(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "1", "--seed", "3", "--output-dir", "out",
        ];
        args.extend_from_slice(extra);
        self.camdl(&args)
    }

    /// Every `Sim` leaf: `(dir, run_id, scenario level label)`, sorted by run_id.
    fn leaves(&self) -> Vec<(PathBuf, String, String)> {
        let mut v: Vec<(PathBuf, String, String)> = run_leaves(&self.path("out").join("sims"))
            .into_iter()
            .map(|dir| {
                let rec: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
                let label = rec["levels"].as_array().unwrap().iter()
                    .find(|l| l["name"] == "scenario")
                    .map(|l| l["label"].as_str().unwrap().to_string())
                    .unwrap();
                (dir, rec["run_id"].as_str().unwrap().to_string(), label)
            })
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    }

    fn ensemble_id(&self) -> String {
        let ens = run_leaves(&self.path("out").join("ensembles"));
        assert_eq!(ens.len(), 1, "one ensemble leaf: {ens:?}");
        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ens[0].join("run.json")).unwrap()).unwrap();
        rec["run_id"].as_str().unwrap().to_string()
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

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what}: exited {:?}; stderr:\n{}", out.status.code(), stderr(out));
}

fn simulated(out: &Output) -> Option<usize> {
    stderr(out)
        .lines()
        .find(|l| l.contains("read from the store; simulated "))
        .map(|l| l.rsplit("simulated ").next().unwrap().trim().parse().unwrap())
}

/// The scenario path segment's label part: the leaf's parent directory is
/// `<label>-<h8>`.
fn scenario_segment_label(leaf: &Path) -> String {
    let seg = leaf.parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
    seg.rsplit_once('-').map(|(l, _)| l.to_string()).unwrap_or(seg)
}

/// Test 10 — `--enable vaccination` labels its arm `(none)+vaccination` in
/// `run.json`, in the leaf path (rendered by the store's path sanitizer, which
/// maps `(`, `)` and `+` to `_`) and in the store listing; `--param beta=0.3`
/// and `--param beta=.3` both give `(none);beta=0.3` and one leaf; plain
/// `simulate` gives `(none)`, in `run.json`, the path and the listing.
#[test]
fn ad_hoc_arms_are_labelled_from_their_flags() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--enable", "vaccination", "--replicates", "2"]), "--enable");
    let leaves = fx.leaves();
    assert_eq!(leaves.len(), 2);
    for (dir, _, label) in &leaves {
        assert_eq!(label, "(none)+vaccination", "run.json level label");
        assert_eq!(scenario_segment_label(dir), "_none__vaccination", "path segment: {}", dir.display());
    }
    let list = fx.camdl(&["list", "out", "--kind", "sim"]);
    assert_ok(&list, "list");
    let listing = format!("{}{}", String::from_utf8_lossy(&list.stdout), stderr(&list));
    assert!(
        listing.contains("(none)+vaccination"),
        "the listing shows the derived label:\n{listing}"
    );

    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--param", "beta=0.3"]), "--param beta=0.3");
    assert_ok(&fx.simulate(&["--param", "beta=.3"]), "--param beta=.3");
    let leaves = fx.leaves();
    assert_eq!(leaves.len(), 1, "`beta=.3` is `beta=0.3`: one label, one leaf: {leaves:?}");
    assert_eq!(leaves[0].2, "(none);beta=0.3");
    assert_eq!(scenario_segment_label(&leaves[0].0), "_none__beta_0.3");

    let fx = Fixture::new();
    assert_ok(&fx.simulate(&[]), "plain");
    let leaves = fx.leaves();
    assert_eq!(leaves[0].2, "(none)");
    assert_eq!(scenario_segment_label(&leaves[0].0), "_none_");
    let list = fx.camdl(&["list", "out", "--kind", "sim"]);
    assert_ok(&list, "list");
    let listing = format!("{}{}", String::from_utf8_lossy(&list.stdout), stderr(&list));
    assert!(listing.contains("(none)"), "the listing shows `(none)`:\n{listing}");
}

/// Test 10 / test 3 — the label is not identity: the preset `vax` and
/// `--enable vaccination` resolve to one delta, so the same leaves (one
/// `run_id` set, found by hash across the two labels, nothing simulated twice)
/// and one ensemble identity.
#[test]
fn a_preset_and_the_flags_that_spell_it_share_leaves_and_ensemble() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--scenario", "vax", "--replicates", "2"]), "preset");
    let preset_leaves = fx.leaves();
    let preset_ens = fx.ensemble_id();
    assert!(preset_leaves.iter().all(|(_, _, l)| l == "vax"));

    // Flags in the same store: every cell a hit on the preset's leaves.
    let run = fx.simulate(&["--enable", "vaccination", "--replicates", "2"]);
    assert_ok(&run, "flags over the preset's store");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves(), preset_leaves, "no second copy under `(none)+vaccination`");
    assert_eq!(fx.ensemble_id(), preset_ens, "one ensemble identity whatever the arm is called");

    // Flags in a fresh store: the same run_ids under the derived label.
    let fy = Fixture::new();
    assert_ok(&fy.simulate(&["--enable", "vaccination", "--replicates", "2"]), "flags, fresh store");
    let flag_leaves = fy.leaves();
    assert_eq!(
        flag_leaves.iter().map(|l| &l.1).collect::<Vec<_>>(),
        preset_leaves.iter().map(|l| &l.1).collect::<Vec<_>>(),
        "the leaf identities are the resolved delta's, unchanged by the label"
    );
    assert!(flag_leaves.iter().all(|(_, _, l)| l == "(none)+vaccination"));
    assert_eq!(fy.ensemble_id(), preset_ens);
}

/// Test 10 — a leaf written under `baseline-<h8>` before this change (the
/// pre-PR binary labelled every ad-hoc arm `baseline`) is a hit for the
/// relabelled run: nothing is simulated and no `(none)+vaccination` copy
/// appears. The pre-PR shape is reproduced on disk by relabelling the
/// scenario directory and its records.
#[test]
fn a_leaf_written_under_the_old_baseline_label_is_a_hit() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--enable", "vaccination", "--replicates", "2"]), "first run");
    let leaves = fx.leaves();
    let scenario_dir = leaves[0].0.parent().unwrap().to_path_buf();
    let seg = scenario_dir.file_name().unwrap().to_string_lossy().into_owned();
    let (_, h8) = seg.rsplit_once('-').unwrap();
    let old_shape = scenario_dir.with_file_name(format!("baseline-{h8}"));
    std::fs::rename(&scenario_dir, &old_shape).unwrap();
    for leaf in run_leaves(&old_shape) {
        let path = leaf.join("run.json");
        let mut rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for level in rec["levels"].as_array_mut().unwrap() {
            if level["name"] == "scenario" {
                level["label"] = serde_json::Value::String("baseline".into());
            }
        }
        std::fs::write(&path, serde_json::to_string_pretty(&rec).unwrap()).unwrap();
    }
    let old_leaves = fx.leaves();
    assert!(old_leaves.iter().all(|(d, _, l)| l == "baseline" && d.starts_with(&old_shape)));

    let dry = fx.simulate(&["--enable", "vaccination", "--replicates", "2", "--dry-run"]);
    assert_ok(&dry, "dry run");
    assert!(stderr(&dry).contains("2 cache hits → read back, 0 misses"), "{}", stderr(&dry));
    let run = fx.simulate(&["--enable", "vaccination", "--replicates", "2"]);
    assert_ok(&run, "relabelled run");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves(), old_leaves, "found by hash; the old-labelled leaves stay the only copy");
    assert!(!scenario_dir.exists(), "no `(none)+vaccination` directory appears");
}
