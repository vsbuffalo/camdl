//! gh#573: a sim leaf's `run_id` must be a function of the scenario's
//! *effective* delta — composed `enable`/`disable`/`set` plus `scale` — not of
//! the preset's own literal fields.
//!
//! Before the fix the `scenario` identity level hashed only a preset's own
//! `enable`/`disable`/`params`, while the value resolver applied composed
//! fields and `scale`. Two presets differing only in `scale`, or only in what
//! they `compose`, received one `run_id` and produced two trajectories.
//!
//! CLI-level on purpose: the defect was in the wiring (which fields reached
//! the hasher), so a hasher-only test would stay green while it was broken.
//! Both store paths are covered — several `--scenario` arms in one invocation
//! (the multi-cell sink) and one arm per invocation (the single-cell path),
//! which build their scenario delta at separate sites.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Pure-death model, run with `mu = 0.5` ([`PARAMS`]). The presets change it
/// only through `scale` and `compose`, never through `set`.
///
/// - `halved` / `doubled`: differ only in `scale` (case a).
/// - `via_half` / `via_double`: own fields empty; differ only in which
///   sub-preset they compose (case b).
/// - `halved` vs `via_half`: the same effective delta (`mu × 0.5`) spelled
///   directly and through `compose` (case c, the negative control).
const MODEL: &str = r#"
time_unit = 'days
compartments { S, D }
parameters { mu : rate in [0.001, 5.0] }
transitions { death : S --> D @ mu * S }
init { S = 1000  D = 0 }
simulate { from = 0 'days  to = 20 'days }

scenarios {
  halved     { scale = { mu = 0.5 } }
  doubled    { scale = { mu = 2.0 } }
  half       { scale = { mu = 0.5 } }
  double     { scale = { mu = 2.0 } }
  via_half   { compose = [half] }
  via_double { compose = [double] }
}
"#;

/// Base parameter values, identical for every scenario.
const PARAMS: &str = "mu = 0.5\n";

/// Write the model and params file into `dir`; returns their paths.
fn write_inputs(dir: &Path) -> (PathBuf, PathBuf) {
    let model = dir.join("decay.camdl");
    let params = dir.join("p.toml");
    std::fs::write(&model, MODEL).unwrap();
    std::fs::write(&params, PARAMS).unwrap();
    (model, params)
}

/// Every directory under `root` holding a `run.json` (the CAS leaves).
fn run_leaves(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join("run.json").is_file() {
            out.push(dir.clone());
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                }
            }
        }
    }
    out
}

/// One leaf's `run_id` and numeric trajectory rows.
#[derive(Debug)]
struct Leaf {
    run_id: String,
    rows: Vec<String>,
}

/// The scenario label of a leaf: the `scenario` path segment is
/// `<label>-<h8>`, the parent of the `seed_N-<h8>` leaf directory.
fn scenario_label(leaf: &Path) -> String {
    let seg = leaf.parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
    seg.rsplit_once('-').map(|(l, _)| l.to_string()).unwrap_or(seg)
}

fn read_leaf(leaf: &Path) -> Leaf {
    let rj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    let run_id = rj["run_id"].as_str().expect("run.json has run_id").to_string();
    let rows = std::fs::read_to_string(leaf.join("traj.tsv"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|l| l.split('\t').next().is_some_and(|f| f.parse::<f64>().is_ok()))
        .map(str::to_string)
        .collect();
    Leaf { run_id, rows }
}

/// Leaves keyed by scenario label.
fn leaves_by_scenario(sims: &Path) -> BTreeMap<String, Leaf> {
    run_leaves(sims).iter().map(|l| (scenario_label(l), read_leaf(l))).collect()
}

/// Run one `simulate --scenario a,b,…` over `scenarios` (seed 1,
/// chain_binomial, dt 1): every arm in one multi-cell invocation.
fn simulate_arms(scenarios: &[&str]) -> BTreeMap<String, Leaf> {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = write_inputs(tmp.path());
    let out = tmp.path().join("out");
    let arms = scenarios.join(",");
    let run = Command::new(&bin)
        .args([
            "simulate", &model.to_string_lossy(),
            "--params", &params.to_string_lossy(),
            "--scenario", &arms,
            "--backend", "chain_binomial",
            "--dt", "1",
            "--seed", "1",
            "--output-dir", &out.to_string_lossy(),
        ])
        .output()
        .expect("spawn");
    assert!(
        run.status.success(),
        "simulate --scenario {arms} failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    leaves_by_scenario(&out.join("sims"))
}

/// Run `simulate --scenario s` once per scenario into one store. Returns the
/// store's leaves by scenario label and each scenario's `-o` mirror rows.
fn simulate_each(scenarios: &[&str]) -> (BTreeMap<String, Leaf>, BTreeMap<String, Vec<String>>) {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let (model, params) = write_inputs(tmp.path());
    let out = tmp.path().join("out");
    for s in scenarios {
        let run = Command::new(&bin)
            .args([
                "simulate", &model.to_string_lossy(),
                "--params", &params.to_string_lossy(),
                "--scenario", s,
                "--backend", "chain_binomial",
                "--dt", "1",
                "--seed", "1",
                "--output-dir", &out.to_string_lossy(),
                "-o", &tmp.path().join(format!("{s}.tsv")).to_string_lossy(),
            ])
            .output()
            .expect("spawn");
        assert!(
            run.status.success(),
            "simulate --scenario {s} failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
    let mirrors = scenarios
        .iter()
        .map(|s| {
            let rows = std::fs::read_to_string(tmp.path().join(format!("{s}.tsv")))
                .unwrap()
                .lines()
                .filter(|l| !l.starts_with('#'))
                .filter(|l| l.split('\t').next().is_some_and(|f| f.parse::<f64>().is_ok()))
                .map(str::to_string)
                .collect();
            (s.to_string(), rows)
        })
        .collect();
    (leaves_by_scenario(&out.join("sims")), mirrors)
}

/// The premise and the claim for a pair that must NOT share an address: the
/// trajectories differ, so the `run_id`s must too.
fn assert_distinct(leaves: &BTreeMap<String, Leaf>, a: &str, b: &str) {
    assert_ne!(
        leaves[a].rows, leaves[b].rows,
        "premise: `{a}` and `{b}` must produce different trajectories"
    );
    assert_ne!(
        leaves[a].run_id, leaves[b].run_id,
        "`{a}` and `{b}` produce different trajectories but share run_id {} — \
         the scenario identity level is not hashing the effective delta (gh#573)",
        leaves[a].run_id
    );
}

/// (a) Scale-only difference re-keys — several arms in one invocation.
#[test]
fn multi_scenario_scale_only_difference_changes_run_id() {
    let leaves = simulate_arms(&["halved", "doubled"]);
    assert_eq!(leaves.len(), 2, "two identities are two leaves: {leaves:?}");
    assert_distinct(&leaves, "halved", "doubled");
}

/// (b) Compose-only difference re-keys — several arms in one invocation. The
/// two presets' own fields are empty; everything they change arrives through
/// `compose`.
#[test]
fn multi_scenario_compose_only_difference_changes_run_id() {
    let leaves = simulate_arms(&["via_half", "via_double"]);
    assert_eq!(leaves.len(), 2, "two identities are two leaves: {leaves:?}");
    assert_distinct(&leaves, "via_half", "via_double");
}

/// (c) Negative control. `halved` scales `mu` by 0.5 directly; `via_half`
/// does it by composing `half`. The effective delta is identical, so the
/// trajectory is identical, and a content address — a function of content,
/// not of spelling — must be identical too. One identity is one leaf: the
/// path carries the scenario *label*, which is provenance, and the store
/// finds a leaf by its hashes (`runid::find_leaf`), so the second arm is a
/// hit on the first arm's leaf rather than a second copy under its own name
/// (proposal 2026-10-24-batch-to-simulate §3 "Arm labels").
#[test]
fn multi_scenario_same_effective_delta_via_compose_shares_run_id() {
    let leaves = simulate_arms(&["halved", "via_half"]);
    assert_eq!(
        leaves.keys().collect::<Vec<_>>(),
        vec!["halved"],
        "the same effective scenario delta, spelled directly vs through compose, \
         hashes to one run_id and so is one leaf — under the label that wrote it \
         first, never a second directory: {leaves:?}"
    );
}

/// (a)+(b)+(c) through the single-cell path — one `simulate --scenario s`
/// per arm — which builds its scenario delta at a separate site from the
/// multi-cell sink.
#[test]
fn simulate_scenario_identity_hashes_effective_delta() {
    let (leaves, mirrors) = simulate_each(&["halved", "doubled", "via_half", "via_double"]);
    assert_distinct(&leaves, "halved", "doubled");
    assert_eq!(
        mirrors["halved"], mirrors["via_half"],
        "premise: the same effective delta must produce the same trajectory"
    );
    assert_ne!(mirrors["via_half"], mirrors["via_double"]);
    // `via_half` and `via_double` ran after `halved` and `doubled`; each found
    // the leaf its identity already had and wrote no second copy, so the store
    // holds exactly the two identities, labelled by the arms that wrote them.
    assert_eq!(
        leaves.keys().collect::<Vec<_>>(),
        vec!["doubled", "halved"],
        "simulate: same effective delta must share a run_id, and so a leaf: {leaves:?}"
    );
}
