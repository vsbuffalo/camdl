//! The label of `simulate`'s scenario folder. A named scenario is labelled by
//! its name; every other arm, whatever `--enable`, `--disable` or `--param`
//! built it, is the implicit arm, labelled `(none)` (proposal
//! `docs/dev/proposals/2026-11-20-run-store-api.md`, "One simulation"). A
//! label is never part of a cell's identity: the arm's identity is the
//! interventions it keeps and the values it runs, so a preset-named arm with
//! the same effect is the same cell, and a cell stored under another label is
//! found by identity and reused.
//!
//! Shells out to the release `camdl` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

mod sim_store_cells;

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
        Command::new(skip_if_missing_binary()).env_remove("CLICOLOR_FORCE")
            .args(args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    fn simulate(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "1", "--seed", "3", "--store", "out",
        ];
        args.extend_from_slice(extra);
        self.camdl(&args)
    }

    /// Every stored cell: `(dir, run_id, scenario folder label)`, sorted by
    /// run_id.
    fn leaves(&self) -> Vec<(PathBuf, String, String)> {
        let mut v: Vec<(PathBuf, String, String)> = sim_store_cells::sim_cells(&self.path("out"))
            .into_iter()
            .map(|c| {
                let label = c.scenario_dir_name();
                (c.dir, c.run_id, label)
            })
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    }

    /// `camdl list --kind sim` over the store, stdout and stderr together.
    fn listing(&self) -> String {
        let list = self.camdl(&["list", "--store", "out", "--kind", "sim"]);
        assert_ok(&list, "list");
        format!("{}{}", String::from_utf8_lossy(&list.stdout), stderr(&list))
    }
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

/// Every arm without a named scenario is the implicit arm, labelled `(none)`
/// (`_none_` in the folder name, where the path sanitizer maps `(` and `)` to
/// `_`): `--enable vaccination`, `--param beta=0.3` and a plain run alike.
/// `--param beta=0.3` and `--param beta=.3` are one cell.
#[test]
fn ad_hoc_arms_are_the_implicit_arm() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--enable", "vaccination", "--replicates", "2"]), "--enable");
    let leaves = fx.leaves();
    assert_eq!(leaves.len(), 2);
    for (dir, _, label) in &leaves {
        assert_eq!(label, "_none_", "path segment: {}", dir.display());
    }
    let listing = fx.listing();
    assert!(listing.contains("(none)"), "the listing shows `(none)`:\n{listing}");

    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--param", "beta=0.3"]), "--param beta=0.3");
    assert_ok(&fx.simulate(&["--param", "beta=.3"]), "--param beta=.3");
    let leaves = fx.leaves();
    assert_eq!(leaves.len(), 1, "`beta=.3` is `beta=0.3`: one label, one cell: {leaves:?}");
    assert_eq!(leaves[0].2, "_none_");
    let listing = fx.listing();
    assert!(listing.contains("(none)"), "the listing shows `(none)`:\n{listing}");

    let fx = Fixture::new();
    assert_ok(&fx.simulate(&[]), "plain");
    let leaves = fx.leaves();
    assert_eq!(leaves[0].2, "_none_");
    let listing = fx.listing();
    assert!(listing.contains("(none)"), "the listing shows `(none)`:\n{listing}");
}

/// Test 10 / test 3 — the label is not identity: the preset `vax` and
/// `--enable vaccination` resolve to one delta, so the same cells (one
/// `run_id` set, found across the two labels, nothing simulated twice).
#[test]
fn a_preset_and_the_flags_that_spell_it_share_leaves() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--scenario", "vax", "--replicates", "2"]), "preset");
    let preset_leaves = fx.leaves();
    assert!(preset_leaves.iter().all(|(_, _, l)| l == "vax"));

    // Flags in the same store: every cell a hit on the preset's cells.
    let run = fx.simulate(&["--enable", "vaccination", "--replicates", "2"]);
    assert_ok(&run, "flags over the preset's store");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves(), preset_leaves, "no second copy under `(none)`");

    // Flags in a fresh store: the same run_ids under the derived label.
    let fy = Fixture::new();
    assert_ok(&fy.simulate(&["--enable", "vaccination", "--replicates", "2"]), "flags, fresh store");
    let flag_leaves = fy.leaves();
    assert_eq!(
        flag_leaves.iter().map(|l| &l.1).collect::<Vec<_>>(),
        preset_leaves.iter().map(|l| &l.1).collect::<Vec<_>>(),
        "the cell identities are the resolved delta's, unchanged by the label"
    );
    assert!(flag_leaves.iter().all(|(_, _, l)| l == "_none_"));
}

/// Test 10 — a cell stored under a scenario folder with another label
/// (`baseline-<h12>`, as an earlier binary labelled every ad-hoc arm) is a hit
/// for the relabelled run: nothing is simulated and no `(none)` copy
/// appears. The shape is reproduced on disk by renaming the scenario
/// folder.
#[test]
fn a_leaf_written_under_the_old_baseline_label_is_a_hit() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--enable", "vaccination", "--replicates", "2"]), "first run");
    let leaves = fx.leaves();
    let scenario_dir = leaves[0].0.parent().unwrap().to_path_buf();
    let seg = scenario_dir.file_name().unwrap().to_string_lossy().into_owned();
    let (_, h) = seg.rsplit_once('-').unwrap();
    let old_shape = scenario_dir.with_file_name(format!("baseline-{h}"));
    std::fs::rename(&scenario_dir, &old_shape).unwrap();
    let old_leaves = fx.leaves();
    assert!(old_leaves.iter().all(|(d, _, l)| l == "baseline" && d.starts_with(&old_shape)));

    let dry = fx.simulate(&["--enable", "vaccination", "--replicates", "2", "--dry-run"]);
    assert_ok(&dry, "dry run");
    assert!(stderr(&dry).contains("2 cache hits → read back, 0 misses"), "{}", stderr(&dry));
    let run = fx.simulate(&["--enable", "vaccination", "--replicates", "2"]);
    assert_ok(&run, "relabelled run");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(fx.leaves(), old_leaves, "found by identity; the old-labelled cells stay the only copy");
    assert!(!scenario_dir.exists(), "no `(none)` directory appears");
}
