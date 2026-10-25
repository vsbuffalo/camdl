//! `camdl simulate --sweep PARAM=GRID` / `--sweep-file FILE`: the design axis
//! (proposal `docs/dev/proposals/2026-10-24-batch-to-simulate.md` §3, §6
//! tests 1, 2, 4, 5, 6).
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

/// SIR with one reported-cases stream whose reporting fraction `rho` is read
/// by the observation model only — the dynamics never see it. Sweeping `rho`
/// with observation output disabled therefore changes nothing but the leaf's
/// identity, which is what makes it the D-B pin: two design points at one
/// replicate share a process seed iff their trajectories are byte-identical.
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

simulate { from = 0 'days  to = 30 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}

scenarios {
  low_beta { set = { beta = 0.2 } }
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

    fn write(&self, name: &str, text: &str) -> String {
        std::fs::write(self.path(name), text).unwrap();
        name.to_string()
    }

    /// `simulate m.camdl --params p.toml --backend chain_binomial --dt 1 --output-dir out <extra…>`.
    fn simulate(&self, extra: &[&str]) -> Output {
        let mut args = vec![
            "simulate", "m.camdl", "--params", "p.toml", "--backend", "chain_binomial",
            "--dt", "1", "--output-dir", "out",
        ];
        args.extend_from_slice(extra);
        Command::new(skip_if_missing_binary())
            .args(&args)
            .current_dir(self.dir.path())
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output()
            .expect("spawn camdl")
    }

    /// Every `Sim` leaf under the output root.
    fn leaves(&self) -> Vec<PathBuf> {
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

fn assert_refused(out: &Output, what: &str, needles: &[&str]) {
    let err = stderr(out);
    assert!(!out.status.success(), "{what}: must be refused, but exited 0; stderr:\n{err}");
    for n in needles {
        assert!(err.contains(n), "{what}: diagnostic lacks {n:?}; stderr:\n{err}");
    }
}

/// The `<label>` of a leaf's `<level>` path segment (`<label>-<h8>`), read
/// from `run.json`'s levels so the test does not parse directory names.
fn level_label(leaf: &Path, level: &str) -> String {
    let rec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    rec["levels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["name"] == level)
        .map(|l| l["label"].as_str().unwrap().to_string())
        .unwrap_or_else(|| panic!("no `{level}` level in {}", leaf.display()))
}

fn process_seed(leaf: &Path) -> u64 {
    let rec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    rec["inputs"]["process_seed"].as_u64().expect("process_seed in inputs")
}

/// Test 1 — the D-B pin. `--sweep rho=0.5,0.9 --replicates 2` writes four
/// leaves labelled `rho=0.5` and `rho=0.9`; the two points share a process seed
/// per replicate, and because `rho` is read by the observation model only
/// (observation output disabled here) their trajectories are byte-identical
/// per replicate — and differ across replicates.
#[test]
fn design_points_share_a_process_seed_per_replicate() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--sweep", "rho=0.5,0.9", "--replicates", "2", "--seed", "11"]), "sweep");
    let leaves = fx.leaves();
    assert_eq!(leaves.len(), 4, "2 points × 2 replicates: {leaves:?}");

    let mut labels: Vec<String> = leaves.iter().map(|l| level_label(l, "params")).collect();
    labels.sort();
    labels.dedup();
    assert_eq!(labels, vec!["rho=0.5", "rho=0.9"], "the params level is labelled by the swept value");

    // Group by process seed: each seed must hold exactly the two points.
    let mut by_seed: std::collections::BTreeMap<u64, Vec<&PathBuf>> = Default::default();
    for l in &leaves {
        by_seed.entry(process_seed(l)).or_default().push(l);
    }
    assert_eq!(by_seed.len(), 2, "two replicates are two process seeds, shared across points: {by_seed:?}");
    let mut trajectories: Vec<Vec<u8>> = Vec::new();
    for (seed, group) in &by_seed {
        assert_eq!(group.len(), 2, "both design points share seed {seed}: {group:?}");
        let a = std::fs::read(group[0].join("traj.tsv")).unwrap();
        let b = std::fs::read(group[1].join("traj.tsv")).unwrap();
        assert_eq!(a, b, "the two points at seed {seed} must be byte-identical (rho is observation-only)");
        trajectories.push(a);
    }
    assert_ne!(trajectories[0], trajectories[1], "the two replicates must differ (the sweep ran twice)");
}

/// `--sweep-file` runs each row as a design point, labelled by its values, with
/// `point_id` the 0-based row; a two-parameter row's label is name-sorted.
#[test]
fn sweep_file_rows_are_design_points() {
    let fx = Fixture::new();
    let f = fx.write("d.tsv", "rho\tbeta\n0.5\t0.3\n0.9\t0.3\n");
    assert_ok(&fx.simulate(&["--sweep-file", &f, "--seed", "3"]), "sweep-file");
    let mut labels: Vec<String> = fx.leaves().iter().map(|l| level_label(l, "params")).collect();
    labels.sort();
    assert_eq!(labels, vec!["beta=0.3_rho=0.5", "beta=0.3_rho=0.9"]);
}

/// Test 5 — every conflict is refused with a message naming it.
#[test]
fn sweep_conflicts_are_refused_by_name() {
    let fx = Fixture::new();
    let design_toml = fx.write("design.toml", "");

    assert_refused(
        &fx.simulate(&["--sweep", "beta=0.2,0.3", "--draws", "uniform", "-n", "2"]),
        "--sweep with --draws",
        &["--sweep", "--draws"],
    );
    assert_refused(
        &fx.simulate(&["--sweep-file", "x.tsv", "--draws", "uniform", "-n", "2"]),
        "--sweep-file with --draws",
        &["--sweep-file", "--draws"],
    );
    assert_refused(
        &fx.simulate(&["--sweep", "beta=0.2,0.3", "--design-from", &design_toml]),
        "--sweep with --design-from",
        &["--sweep", "--design-from"],
    );
    assert_refused(
        &fx.simulate(&["--sweep", "beta=0.2,0.3", "--param", "beta=0.3"]),
        "--param on a swept parameter",
        &["parameter `beta` is both swept and pinned by `--param beta=0.3`", "would not vary `beta`"],
    );
    assert_refused(
        &fx.simulate(&["--sweep", "beta=0.2", "--sweep", "beta=0.3"]),
        "a repeated swept name",
        &["--sweep names the same parameter more than once"],
    );
    assert_refused(
        &fx.simulate(&["--sweep", "zeta=0.2,0.3"]),
        "an unknown swept name",
        &["--sweep names parameter(s) the model does not declare: zeta"],
    );
    // A scenario that sets the swept parameter: the engine's coordinate guard
    // (gh#572), worded as for `--draws FILE`; refused before any leaf is written,
    // on the run and on the dry run alike.
    for extra in [&[][..], &["--dry-run"][..]] {
        let mut args = vec!["--sweep", "beta=0.2,0.3", "--scenario", "low_beta"];
        args.extend_from_slice(extra);
        assert_refused(
            &fx.simulate(&args),
            "a scenario setting the swept parameter",
            &[
                "error: parameter `beta` is controlled by both the sweep and scenario `low_beta`",
                "  sweep:     beta = [0.2, 0.3]",
                "  scenario:  set beta = 0.2",
            ],
        );
    }
    // The same guard over a `--sweep-file`.
    let f = fx.write("beta.tsv", "beta\n0.2\n0.3\n");
    assert_refused(
        &fx.simulate(&["--sweep-file", &f, "--scenario", "low_beta"]),
        "a scenario setting a sweep-file parameter",
        &["error: parameter `beta` is controlled by both the sweep and scenario `low_beta`"],
    );
    assert!(fx.leaves().is_empty(), "a refused run writes no leaf");
}

/// Test 6 — `--sweep-file` refusals: an unknown column (with the parameter
/// list), zero rows, a duplicate row (both line numbers named).
#[test]
fn sweep_file_refusals() {
    let fx = Fixture::new();
    let unknown = fx.write("u.tsv", "beta\tzeta\n0.2\t1\n");
    assert_refused(
        &fx.simulate(&["--sweep-file", &unknown]),
        "unknown column",
        &["column(s) `zeta` name no model parameter", "this model declares: beta, gamma, rho"],
    );
    let empty = fx.write("z.tsv", "beta\n");
    assert_refused(&fx.simulate(&["--sweep-file", &empty]), "zero rows", &["zero rows"]);
    let dup = fx.write("d.tsv", "beta\n0.2\n0.3\n0.2\n");
    assert_refused(
        &fx.simulate(&["--sweep-file", &dup]),
        "duplicate row",
        &["line 4 repeats the design point of line 2 (beta=0.2)"],
    );
    assert!(fx.leaves().is_empty(), "a refused run writes no leaf");
}

/// The resolved design lives in the ensemble's `run.json`: the swept
/// parameters with their values and `n_points`, in place of `n_draws`.
#[test]
fn ensemble_record_carries_the_resolved_design() {
    let fx = Fixture::new();
    assert_ok(&fx.simulate(&["--sweep", "rho=0.5,0.9", "--sweep", "beta=0.3", "--replicates", "2"]), "sweep");
    let ens = run_leaves(&fx.path("out").join("ensembles"));
    assert_eq!(ens.len(), 1, "one ensemble leaf: {ens:?}");
    let rec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ens[0].join("run.json")).unwrap()).unwrap();
    let inputs = &rec["inputs"];
    assert_eq!(inputs["n_cells"], 4);
    assert_eq!(inputs["n_points"], 2);
    assert!(inputs.get("n_draws").is_none(), "a sweep job reports n_points, not n_draws: {inputs}");
    assert_eq!(
        inputs["design"],
        serde_json::json!({
            "source": "sweep",
            "n_points": 2,
            "params": [
                { "name": "beta", "values": [0.3] },
                { "name": "rho", "values": [0.5, 0.9] },
            ],
        }),
        "the design is recorded name-sorted with its values: {inputs}"
    );
}
