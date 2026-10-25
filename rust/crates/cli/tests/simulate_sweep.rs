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

quantities {
  peak = max(I / N)
}

scenarios {
  low_beta   { set = { beta = 0.2 } }
  high_gamma { set = { gamma = 0.3 } }
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

/// The header and data rows of a camdl TSV (version line dropped).
fn table(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<String> = lines.next().expect("header").split('\t').map(str::to_string).collect();
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    (header, rows)
}

fn column<'a>(header: &[String], rows: &'a [Vec<String>], name: &str) -> Vec<&'a str> {
    let i = header.iter().position(|h| h == name).unwrap_or_else(|| panic!("no column `{name}` in {header:?}"));
    rows.iter().map(|r| r[i].as_str()).collect()
}

fn distinct(v: &[&str]) -> Vec<String> {
    let mut d: Vec<String> = v.iter().map(|s| s.to_string()).collect();
    d.sort();
    d.dedup();
    d
}

/// Test 2 — the combined table of a sweep job carries `point_id`,
/// `replicate`, `sweep:rho` and `seed` and no `draw`; `sweep:rho` round-trips
/// `0.1234567` exactly; the two cells of one replicate carry the same `seed`;
/// `point_id` and `replicate` count from 0.
#[test]
fn combined_table_of_a_sweep_job_has_point_id_sweep_and_seed_columns() {
    let fx = Fixture::new();
    assert_ok(
        &fx.simulate(&["--sweep", "rho=0.1234567,0.5", "--replicates", "2", "--seed", "11", "-o", "c.tsv"]),
        "sweep",
    );
    let (header, rows) = table(&fx.path("c.tsv"));
    assert_eq!(
        &header[..7],
        ["replicate", "point_id", "sweep:rho", "seed", "t", "t_start", "t_stop"],
        "coordinate columns lead the time columns: {header:?}"
    );
    assert!(!header.iter().any(|h| h == "draw"), "a sweep job writes no `draw` column: {header:?}");
    assert_eq!(distinct(&column(&header, &rows, "sweep:rho")), vec!["0.1234567", "0.5"], "shortest round-trip");
    assert_eq!(distinct(&column(&header, &rows, "point_id")), vec!["0", "1"], "point_id counts from 0");
    assert_eq!(distinct(&column(&header, &rows, "replicate")), vec!["0", "1", "2", "3"], "replicate counts from 0");

    // Pairing: two distinct seeds in all, each carried by both design points.
    let seeds = column(&header, &rows, "seed");
    let points = column(&header, &rows, "point_id");
    assert_eq!(distinct(&seeds).len(), 2, "two replicates are two seeds, shared across points");
    for seed in distinct(&seeds) {
        let with: Vec<String> = distinct(
            &seeds.iter().zip(&points).filter(|(s, _)| **s == seed).map(|(_, p)| *p).collect::<Vec<_>>(),
        );
        assert_eq!(with, vec!["0", "1"], "seed {seed} must appear under both design points");
    }
}

/// Test 4 — `--sweep gamma=… --sweep rho=…` and the flags in the other order
/// give the same `point_id`s and the same combined file (the product is sorted
/// by parameter name, the first name varying slowest).
#[test]
fn point_ids_do_not_depend_on_flag_order() {
    let fx = Fixture::new();
    assert_ok(
        &fx.simulate(&["--sweep", "rho=0.5,0.9", "--sweep", "gamma=0.1,0.2", "--seed", "5", "-o", "a.tsv"]),
        "rho then gamma",
    );
    assert_ok(
        &fx.simulate(&["--sweep", "gamma=0.1,0.2", "--sweep", "rho=0.5,0.9", "--seed", "5", "-o", "b.tsv"]),
        "gamma then rho",
    );
    let a = std::fs::read(fx.path("a.tsv")).unwrap();
    assert_eq!(a, std::fs::read(fx.path("b.tsv")).unwrap(), "flag order must not change the combined file");
    let (header, rows) = table(&fx.path("a.tsv"));
    assert_eq!(&header[..6], ["replicate", "point_id", "sweep:gamma", "sweep:rho", "seed", "t"]);
    // point_id 0 is (gamma=0.1, rho=0.5), 1 is (gamma=0.1, rho=0.9): gamma varies slowest.
    let pid = column(&header, &rows, "point_id");
    let g = column(&header, &rows, "sweep:gamma");
    let r = column(&header, &rows, "sweep:rho");
    let at = |id: &str| -> (String, String) {
        let i = pid.iter().position(|p| *p == id).unwrap();
        (g[i].to_string(), r[i].to_string())
    };
    assert_eq!(at("0"), ("0.1".into(), "0.5".into()));
    assert_eq!(at("1"), ("0.1".into(), "0.9".into()));
    assert_eq!(at("2"), ("0.2".into(), "0.5".into()));
    assert_eq!(at("3"), ("0.2".into(), "0.9".into()));
}

/// Test 8 — the observation tables of a sweep job carry `point_id`,
/// `sweep:rho` and `seed` and no `draw`.
#[test]
fn observation_tables_of_a_sweep_job_carry_the_design_columns() {
    let fx = Fixture::new();
    assert_ok(
        &fx.simulate(&["--sweep", "rho=0.5,0.9", "--replicates", "2", "--seed", "11", "--obs-only-dir", "synth"]),
        "sweep with --obs-only-dir",
    );
    let (header, rows) = table(&fx.path("synth").join("cases.tsv"));
    assert_eq!(header, ["replicate", "point_id", "sweep:rho", "seed", "time", "cases"]);
    assert_eq!(distinct(&column(&header, &rows, "point_id")), vec!["0", "1"]);
    assert_eq!(distinct(&column(&header, &rows, "sweep:rho")), vec!["0.5", "0.9"]);
    assert_eq!(distinct(&column(&header, &rows, "replicate")), vec!["0", "1", "2", "3"]);
    assert_eq!(distinct(&column(&header, &rows, "seed")).len(), 2, "seeds pair the points");
    // The single-file form carries the same coordinates.
    assert_ok(
        &fx.simulate(&["--sweep", "rho=0.5,0.9", "--replicates", "2", "--seed", "11", "--obs-only", "all.tsv"]),
        "sweep with --obs-only",
    );
    let (header, _) = table(&fx.path("all.tsv"));
    assert_eq!(header, ["replicate", "point_id", "sweep:rho", "seed", "time", "cases"]);
}

/// Test 7 — quantities under a sweep are banded per (design point, scenario),
/// never across points; one replicate per point gives a point-mode row per
/// group.
#[test]
fn quantities_are_grouped_per_design_point_and_scenario() {
    // Banded: 2 points × 2 scenarios, 3 replicates each → 4 groups of 3.
    let fx = Fixture::new();
    assert_ok(
        &fx.simulate(&[
            "--sweep", "rho=0.5,0.9", "--scenario", "low_beta,high_gamma", "--replicates", "3",
            "--seed", "2", "--quantities-out", "q",
        ]),
        "banded sweep",
    );
    let (header, rows) = table(&fx.path("q").join("quantities").join("peak.tsv"));
    assert_eq!(&header[..2], ["scenario", "sweep:rho"], "{header:?}");
    assert!(header.iter().any(|h| h == "n_draws"), "banded: {header:?}");
    assert_eq!(rows.len(), 4, "one band per (point, scenario): {rows:?}");
    let groups: std::collections::BTreeSet<(String, String)> = rows
        .iter()
        .map(|r| (r[0].clone(), r[1].clone()))
        .collect();
    assert_eq!(groups.len(), 4, "each (scenario, point) once: {groups:?}");
    assert!(column(&header, &rows, "n_draws").iter().all(|n| *n == "3"), "a band over the 3 replicates of its group: {rows:?}");

    // Point mode: one replicate per point → one point-mode row per group.
    let fx = Fixture::new();
    assert_ok(
        &fx.simulate(&["--sweep", "rho=0.5,0.9", "--seed", "2", "--quantities-out", "q"]),
        "point-mode sweep",
    );
    let (header, rows) = table(&fx.path("q").join("quantities").join("peak.tsv"));
    assert_eq!(header, ["sweep:rho", "value"], "{header:?}");
    assert_eq!(distinct(&column(&header, &rows, "sweep:rho")), vec!["0.5", "0.9"]);
    assert_eq!(rows.len(), 2);
}
