//! Integration tests for the `camdl pfilter` content-addressed writer +
//! reader (gh#147 M3.3). Drives the release binary end-to-end and asserts the
//! write → read → visible contract: an eval writes a `Pfilter` leaf under
//! `pfilters/`, and `camdl show`/`cat`/`list` surface its loglik + scored
//! point. Plus the identity guard: distinct scored points land at distinct
//! content-addressed leaves.
//!
//! Skipped when the release binary or `camdlc` isn't present (mirrors the
//! rest of the integration suite).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_pfilter_cas_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// SIR-with-Poisson-cases fixture + a tiny observed series.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 10 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    let data = dir.join("cases.tsv");
    std::fs::write(&data, "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    (ir_path, data)
}

fn run_pfilter(bin: &Path, out_root: &Path, ir: &Path, data: &Path, beta: f64) -> std::process::Output {
    run_pfilter_reps(bin, out_root, ir, data, beta, 1)
}

fn run_pfilter_reps(
    bin: &Path, out_root: &Path, ir: &Path, data: &Path, beta: f64, replicates: usize,
) -> std::process::Output {
    Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args([
            "pfilter", &ir.to_string_lossy(),
            "--param", &format!("beta={}", beta),
            "--param", "gamma=0.1", "--param", "N0=1000",
            "--data", &data.to_string_lossy(), "--obs", "cases",
            "--particles", "30", "--dt", "1", "--seed", "1",
            "--replicates", &replicates.to_string(),
        ])
        .output().expect("spawn camdl pfilter")
}

/// The single `Pfilter` leaf dir under `<out_root>/pfilters/` (run.json kind
/// = "pfilter"), and its `run_id`.
fn find_pfilter_leaf(out_root: &Path) -> (PathBuf, String) {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("pfilter") {
                    let rid = v.get("run_id").and_then(|r| r.as_str()).unwrap_or("").to_string();
                    out.push((dir.to_path_buf(), rid));
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), out); } }
        }
    }
    let mut found = Vec::new();
    walk(&out_root.join("pfilters"), &mut found);
    assert_eq!(found.len(), 1, "expected exactly one pfilter leaf, got {:?}", found);
    found.into_iter().next().unwrap()
}

fn camdl_read(bin: &Path, out_root: &Path, args: &[&str]) -> String {
    let out = Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args).output().expect("spawn camdl reader");
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Write → read → visible: a pfilter eval writes a content-addressed leaf, and
/// `show`/`cat`/`list` surface the loglik + scored point. Non-vacuous — it
/// runs a real particle filter and parses the recorded result back out;
/// `find_pfilter_leaf` asserts a leaf actually exists (an early-return on the
/// missing-binary guard would find none and panic, not pass silently).
#[test]
fn pfilter_eval_round_trips_through_reader() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("roundtrip");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_pfilter(&bin, &out_root, &ir, &data, 0.3);
    assert!(output.status.success(), "pfilter failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));
    // stdout contract preserved: a finite loglik on stdout.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim().parse::<f64>().is_ok(),
        "pfilter must still print a loglik to stdout, got: {:?}", stdout);

    // Write side: the leaf carries run.json + loglik.toml, and the recorded
    // loglik is finite + matches the stdout value.
    let (leaf, rid) = find_pfilter_leaf(&out_root);
    assert!(leaf.join("run.json").is_file() && leaf.join("loglik.toml").is_file(),
        "leaf must hold run.json + loglik.toml: {}", leaf.display());
    let rec: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    let recorded = rec["inputs"]["loglik"].as_f64().expect("recorded loglik must be finite");
    let printed: f64 = stdout.trim().parse().unwrap();
    // stdout prints the loglik at 4 decimals (`{:.4}`); the recorded value is
    // full f64, so compare at the printed precision.
    assert!((recorded - printed).abs() < 1e-3,
        "recorded loglik {} must match stdout {} (to 4 decimals)", recorded, printed);
    // The scored point is recorded.
    let params = rec["inputs"]["params"].as_array().expect("params recorded");
    assert!(params.iter().any(|p| p[0].as_str() == Some("beta")),
        "scored point must record beta: {:?}", params);

    // Read side: `show` surfaces the levels + loglik + scored point.
    let shown = camdl_read(&bin, &out_root, &["show", &rid[..12]]);
    for needle in ["pfilter", "model", "config", "params", "seed", "loglik", "scored point", "beta"] {
        assert!(shown.contains(needle), "show must surface {:?}. Got:\n{}", needle, shown);
    }
    // `cat` returns the loglik.toml.
    let catted = camdl_read(&bin, &out_root, &["cat", &rid[..12]]);
    assert!(catted.contains("loglik ="), "cat must return loglik.toml. Got:\n{}", catted);
    // `list --kind pfilter` shows the eval.
    let listed = camdl_read(&bin, &out_root, &["list", "--kind", "pfilter"]);
    assert!(listed.contains("pfilters") && listed.contains(&rid[..8]),
        "list must surface the pfilter eval. Got:\n{}", listed);
}

/// Identity guard: distinct scored points content-address to distinct leaves
/// (collision-free on the params axis); the same point is stable.
#[test]
fn distinct_points_distinct_leaves() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("identity");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    assert!(run_pfilter(&bin, &out_root, &ir, &data, 0.3).status.success());
    let (leaf_a, rid_a) = find_pfilter_leaf_first(&out_root);
    // A different beta → a distinct leaf.
    assert!(run_pfilter(&bin, &out_root, &ir, &data, 0.5).status.success());
    // Re-running the first point must NOT create a third leaf (stable).
    assert!(run_pfilter(&bin, &out_root, &ir, &data, 0.3).status.success());

    let leaves = all_pfilter_leaves(&out_root);
    assert_eq!(leaves.len(), 2,
        "two distinct points (β=0.3, β=0.5) must yield exactly 2 leaves (the \
         β=0.3 rerun is idempotent); got {:?}", leaves);
    assert!(leaves.iter().any(|(_, r)| r == &rid_a),
        "the β=0.3 leaf {} must persist across the rerun", rid_a);
    let _ = leaf_a;
}

/// Priority-zero identity guard (the n_trajectories collision class, M3.2):
/// `--replicates N` averages N PF runs into the stored loglik, so the
/// replicate count changes the stored value and MUST be in the key. Two runs
/// identical except `--replicates 1` vs `3` must land at distinct content-
/// addressed leaves — the same path would silently return a wrong-count
/// loglik (the second run cache-colliding the first).
#[test]
fn replicate_count_is_in_the_identity() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("reps_identity");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    assert!(run_pfilter_reps(&bin, &out_root, &ir, &data, 0.3, 1).status.success());
    assert!(run_pfilter_reps(&bin, &out_root, &ir, &data, 0.3, 3).status.success());

    let leaves = all_pfilter_leaves(&out_root);
    assert_eq!(leaves.len(), 2,
        "--replicates 1 vs 3 (identical otherwise) must be distinct CAS leaves — \
         the replicate count changes the stored loglik, so it must be in the \
         identity. Same path = silent wrong-count cache (the n_trajectories \
         collision class). Got {:?}", leaves);
}

/// `camdl label` covers the pfilter kind (gh#147 item C): labelling a pfilter
/// run by its `run_id` prefix writes `provenance.label` on the leaf's
/// `run.json`, and `show`/`list` then surface that label. Before item C,
/// `cmd_label` only resolved sims/fits/profiles, so this errored with
/// "no run found" even though `show` could resolve the same leaf.
#[test]
fn label_works_on_pfilter_runs() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("label_pfilter");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    assert!(run_pfilter(&bin, &out_root, &ir, &data, 0.3).status.success(),
        "pfilter must succeed");
    let (leaf, rid) = find_pfilter_leaf(&out_root);

    // Label by run_id prefix. `--root` is honoured here; the env path is
    // exercised by the reader helpers, which set CAMDL_OUTPUT_DIR.
    let labelled = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["label", &rid[..8], "baseline-pfilter",
               "--root", &out_root.to_string_lossy()])
        .output().expect("spawn camdl label");
    assert!(labelled.status.success(),
        "label on a pfilter run must succeed (item C). stderr=\n{}",
        String::from_utf8_lossy(&labelled.stderr));

    // Write side: the label persists on the leaf's RunRecord provenance.
    let rec: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    assert_eq!(rec["provenance"]["label"].as_str(), Some("baseline-pfilter"),
        "label must persist on the pfilter leaf's RunRecord.provenance.label. \
         got: {:?}", rec["provenance"]);

    // Read side: show + list surface the label.
    let shown = camdl_read(&bin, &out_root, &["show", &rid[..12]]);
    assert!(shown.contains("baseline-pfilter"),
        "show must surface the new label. Got:\n{}", shown);
    let listed = camdl_read(&bin, &out_root, &["list", "--kind", "pfilter"]);
    assert!(listed.contains("baseline-pfilter"),
        "list --kind pfilter must surface the new label. Got:\n{}", listed);
}

fn all_pfilter_leaves(out_root: &Path) -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&b) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("pfilter") {
                    out.push((dir.to_path_buf(),
                        v.get("run_id").and_then(|r| r.as_str()).unwrap_or("").to_string()));
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), out); } }
        }
    }
    let mut v = Vec::new();
    walk(&out_root.join("pfilters"), &mut v);
    v
}

fn find_pfilter_leaf_first(out_root: &Path) -> (PathBuf, String) {
    all_pfilter_leaves(out_root).into_iter().next().expect("at least one pfilter leaf")
}
