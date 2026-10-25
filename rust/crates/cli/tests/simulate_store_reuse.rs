//! Multi-cell `camdl simulate` re-uses cells the store already holds
//! (proposal `docs/dev/proposals/2026-10-24-batch-to-simulate.md` §3
//! "Re-using cells the store already holds", "Dry run"; §6 test 9).
//!
//! A committed cell is skipped and its `traj.tsv` is read back for the
//! combined file; `--force` recomputes; `--dry-run` reports every cell as a
//! hit or a miss under the exact rule: a hit requires the leaf's own files and,
//! when observations are requested, the exact `obs/<hash8>-<obs_seed>/`
//! subtree with every stream file and its `obs.json`. A leaf holding another
//! cadence's subtree is a miss. A `--quantities-out` run simulates every cell.
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
  beta  : rate        in [0.01, 5.0]
  gamma : rate        in [0.01, 5.0]
  rho   : probability in [0.001, 1.0]
}

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init { S = 990  I = 10  R = 0 }

simulate { from = 0 'days  to = 20 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}

quantities {
  peak = max(I / N)
}
"#;

const PARAMS: &str = "beta = 0.4\ngamma = 0.2\nrho = 0.5\n";

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

    /// A three-replicate run of the fixture into `out`, plus `extra`.
    fn simulate(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "1", "--seed", "4", "--replicates", "3", "--output-dir", "out",
        ];
        args.extend_from_slice(extra);
        Command::new(skip_if_missing_binary())
            .args(&args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    fn sim_leaves(&self) -> Vec<PathBuf> {
        let mut v = run_leaves(&self.path("out").join("sims"));
        v.sort();
        v
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

/// The `cached` line's "simulated N" count, or `None` when every cell ran.
fn simulated(out: &Output) -> Option<usize> {
    stderr(out)
        .lines()
        .find(|l| l.contains("read from the store; simulated "))
        .map(|l| l.rsplit("simulated ").next().unwrap().trim().parse().unwrap())
}

/// The `obs/<hash8>-<seed>` subtree names under a leaf.
fn obs_subtrees(leaf: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(leaf.join("obs"))
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// A second identical multi-cell run simulates zero cells and produces the
/// same combined file; `--force` recomputes them; `--dry-run` reports every
/// cell as a hit.
#[test]
fn a_second_identical_run_reads_every_cell_back() {
    let fx = Fixture::new();
    let first = fx.simulate(&["-o", "a.tsv"]);
    assert_ok(&first, "first run");
    assert_eq!(simulated(&first), None, "nothing in the store yet: every cell runs; stderr:\n{}", stderr(&first));
    assert_eq!(fx.sim_leaves().len(), 3);
    let a = std::fs::read(fx.path("a.tsv")).unwrap();

    let dry = fx.simulate(&["--dry-run"]);
    assert_ok(&dry, "dry run");
    let err = stderr(&dry);
    assert_eq!(err.matches("[hit ]").count(), 3, "every cell a hit:\n{err}");
    assert!(err.contains("3 cache hits → read back, 0 misses"), "{err}");
    assert_eq!(fx.sim_leaves().len(), 3, "a dry run writes nothing");

    let second = fx.simulate(&["-o", "b.tsv"]);
    assert_ok(&second, "second run");
    assert_eq!(simulated(&second), Some(0), "every cell read back; stderr:\n{}", stderr(&second));
    assert_eq!(std::fs::read(fx.path("b.tsv")).unwrap(), a, "the combined file is reproduced byte for byte");
    assert_eq!(fx.sim_leaves().len(), 3);

    let forced = fx.simulate(&["-o", "c.tsv", "--force"]);
    assert_ok(&forced, "forced run");
    assert_eq!(simulated(&forced), None, "--force recomputes every cell; stderr:\n{}", stderr(&forced));
    assert_eq!(std::fs::read(fx.path("c.tsv")).unwrap(), a, "a recompute reproduces the same bytes");
    assert!(fx.path("out").join(".quarantine").is_dir(), "the displaced leaves were quarantined, not deleted");
}

/// Observations make the hit rule exact: a leaf without the requested
/// `obs/<hash8>-<obs_seed>/` subtree — or holding another cadence's — is
/// re-simulated and gains the requested subtree; the dataset is written; a
/// further identical run is then all hits and still writes the dataset.
#[test]
fn an_observation_request_re_simulates_cells_lacking_the_exact_subtree() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&[]), "trajectory-only run");
    let leaves = fx.sim_leaves();
    assert_eq!(leaves.len(), 3);
    assert!(leaves.iter().all(|l| obs_subtrees(l).is_empty()), "no observations were requested");

    // Requested at a 2-day cadence: every leaf lacks that subtree → all re-simulated.
    let dry = fx.simulate(&["--obs-only-dir", "synth2", "--emit-every", "2", "--dry-run"]);
    assert_ok(&dry, "dry run with observations");
    assert!(stderr(&dry).contains("0 cache hits → read back, 3 misses"), "{}", stderr(&dry));
    let run = fx.simulate(&["--obs-only-dir", "synth2", "--emit-every", "2"]);
    assert_ok(&run, "obs run, 2-day cadence");
    assert_eq!(simulated(&run), None, "no leaf held the subtree: every cell ran; stderr:\n{}", stderr(&run));
    assert!(fx.path("synth2").join("cases.tsv").is_file(), "the dataset is written");
    let two_day: Vec<Vec<String>> = fx.sim_leaves().iter().map(|l| obs_subtrees(l)).collect();
    assert!(two_day.iter().all(|s| s.len() == 1), "one subtree per leaf: {two_day:?}");

    // The model's own daily cadence addresses a DIFFERENT subtree: a leaf
    // holding the 2-day one is a miss, not a hit on "some obs directory".
    let dry = fx.simulate(&["--obs-only-dir", "synth1", "--dry-run"]);
    assert_ok(&dry, "dry run, daily cadence");
    assert!(stderr(&dry).contains("0 cache hits → read back, 3 misses"), "another cadence's subtree is not a hit:\n{}", stderr(&dry));
    let run = fx.simulate(&["--obs-only-dir", "synth1"]);
    assert_ok(&run, "obs run, daily cadence");
    assert_eq!(simulated(&run), None, "every cell re-simulated for the daily subtree; stderr:\n{}", stderr(&run));
    let both: Vec<Vec<String>> = fx.sim_leaves().iter().map(|l| obs_subtrees(l)).collect();
    assert!(both.iter().all(|s| s.len() == 2), "both cadences now coexist under each leaf: {both:?}");
    let daily = std::fs::read(fx.path("synth1").join("cases.tsv")).unwrap();

    // Third run, same flag: all hits, and the dataset is written from the store.
    let dry = fx.simulate(&["--obs-only-dir", "synth1b", "--dry-run"]);
    assert_ok(&dry, "dry run, third");
    assert!(stderr(&dry).contains("3 cache hits → read back, 0 misses"), "{}", stderr(&dry));
    let run = fx.simulate(&["--obs-only-dir", "synth1b"]);
    assert_ok(&run, "third run");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(
        std::fs::read(fx.path("synth1b").join("cases.tsv")).unwrap(), daily,
        "the dataset read back from the store equals the one the simulation wrote"
    );
    // And the stream file alongside the trajectory, same bytes.
    let run = fx.simulate(&["--obs-dir", "synth1c", "-o", "t.tsv"]);
    assert_ok(&run, "obs-dir run");
    assert_eq!(simulated(&run), Some(0), "stderr:\n{}", stderr(&run));
    assert_eq!(std::fs::read(fx.path("synth1c").join("cases.tsv")).unwrap(), daily);
}

/// A `--quantities-out` run simulates every cell — the quantities are a
/// regenerated sidecar over the in-memory trajectory — and the dry run says so.
#[test]
fn a_quantities_run_recomputes_every_cell() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&[]), "first run");
    let dry = fx.simulate(&["--quantities-out", "q", "--dry-run"]);
    assert_ok(&dry, "dry run with quantities");
    assert!(
        stderr(&dry).contains("3 cache hits, 0 misses; quantities requested: 3 cells recomputed"),
        "{}", stderr(&dry)
    );
    let run = fx.simulate(&["--quantities-out", "q"]);
    assert_ok(&run, "quantities run");
    assert_eq!(simulated(&run), None, "every cell ran; stderr:\n{}", stderr(&run));
    assert!(fx.path("q").join("quantities").join("peak.tsv").is_file());
}

/// The pre-existing defect commit `c42f5665` fixed, pinned on its own: a leaf
/// first committed without observations declared no `obs` child in its
/// `run.json`; a later observation run deduped at commit (record unchanged)
/// and wrote `obs/<A>/` beside it, so the leaf failed the exact-set check as
/// `Stale(OrphanFiles)`, and the next observation run at another cadence
/// reclaimed the directory — wiping `obs/<A>/` before writing `obs/<B>/`.
/// Sequence: obs-free run; obs run at cadence A; obs run at cadence B; both
/// subtrees must still be present and both declared under `children.obs`.
/// Red on the binary before the fix (one subtree survives), green after.
#[test]
fn an_earlier_cadences_subtree_survives_a_later_one() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&[]), "obs-free run");
    assert_ok(&fx.simulate(&["--obs-only-dir", "a", "--emit-every", "2"]), "obs run, cadence A");
    assert_ok(&fx.simulate(&["--obs-only-dir", "b"]), "obs run, cadence B");
    let leaves = fx.sim_leaves();
    assert_eq!(leaves.len(), 3);
    for leaf in &leaves {
        let subtrees = obs_subtrees(leaf);
        assert_eq!(subtrees.len(), 2, "both cadences under {}: {subtrees:?}", leaf.display());
        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
        let declared = rec["children"]["obs"].as_array().map(|a| a.len()).unwrap_or(0);
        assert_eq!(declared, 2, "run.json declares both obs children under {}: {}", leaf.display(), rec["children"]);
    }
}
