//! Acceptance tests for gh#241 PR E2: `batch run` `[design.*]` blocks write
//! through the SAME content-addressed `ArtifactKind::Sim` path the normal
//! batch/sweep cells use (via `CasSink` + `engine::run_job`), not the old
//! ad-hoc `traj.tsv` + `{"design_point_index",...}` marker store.
//!
//! These are CLI-level: they shell out to the built `camdl` binary and
//! exercise the real compile → resolve → simulate → CAS commit pipeline.
//!
//! Acceptance bar:
//!   1. A partial/incomplete leaf is NOT a cache hit — only a `Completed` CAS
//!      leaf counts, so an aborted/`Running` cell re-runs.
//!   2. A design cell commits as `ArtifactKind::Sim` (`run.json.kind == "sim"`,
//!      under the canonical `sims/` tree).
//!   3. (unit, in `batch.rs`) an identical design cell and a normal sim cell
//!      resolve to the same `run_id`/path — the dedupe proof.
//!   4. Dry-run reports a hit using the real CAS path (resolve + `store.lookup`),
//!      not the legacy marker.

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

/// A small stochastic model whose only parameter, `mu`, is swept by the design
/// block. Chain-binomial so the run is deterministic at a fixed seed.
fn write_design_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S }

parameters {
  mu : rate in [0.001, 10.0]
}

init { S = 1000 }

transitions {
  death : S -->   @ mu * S
}

simulate { from = 0 'days  to = 20 'days }
"#;
    std::fs::write(path, src).unwrap();
}

/// A `batch.toml` with a `[design.NAME]` block over `mu`.
fn write_design_batch(path: &Path, model: &Path, output: &Path, n: usize, seeds: &[u64]) {
    let seed_list = seeds
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    std::fs::write(
        path,
        format!(
            r#"
[config]
model = "{model}"
output_dir = "{out}"
backend = "chain_binomial"
dt = 1
seeds = {{ list = [{seeds}] }}
parallel = 1

[design.sens]
method = "random"
n = {n}
parameters.mu = {{ range = {{ min = 0.05, max = 0.5 }} }}
"#,
            model = model.display(),
            out = output.display(),
            n = n,
            seeds = seed_list,
        ),
    )
    .unwrap();
}

/// Every directory at any depth under `root` that holds a `run.json` (a CAS
/// leaf). The factored sim path is 5 levels deep.
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

fn run_batch(bin: &Path, batch: &Path) -> std::process::Output {
    Command::new(bin)
        .args(["batch", "run", &batch.to_string_lossy()])
        .output()
        .expect("spawn")
}

/// (2) A design cell commits as `ArtifactKind::Sim`: every leaf under the
/// canonical `<output>/sims/` tree has `run.json.kind == "sim"`, carries a
/// `traj.tsv`, and is `completed`. NO `designs/<name>/sims/` ad-hoc tree, and
/// NO legacy `{"design_point_index",...}` marker.
#[test]
fn design_cells_commit_as_sim_kind_under_sims_tree() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("pd.camdl");
    let output = tmp.path().join("output");
    write_design_model(&model);
    let batch = tmp.path().join("batch.toml");
    write_design_batch(&batch, &model, &output, 3, &[1]);

    let run = run_batch(&bin, &batch);
    assert!(
        run.status.success(),
        "design batch run failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    // Sim leaves live under the canonical sims/ tree (shared with normal sims).
    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 3, "expected 3 design cells (n=3, 1 seed), got {:?}", leaves);

    for leaf in &leaves {
        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap())
                .unwrap();
        assert_eq!(
            rec["kind"].as_str(),
            Some("sim"),
            "a design cell must commit as ArtifactKind::Sim; run.json = {rec}"
        );
        assert_eq!(
            rec["status"].as_str(),
            Some("completed"),
            "a committed design cell must be Completed; run.json = {rec}"
        );
        // The canonical Sim leaf carries traj.tsv; the old marker carried a
        // bare `{"design_point_index"...}` and a hand-written traj.tsv.
        assert!(leaf.join("traj.tsv").exists(), "design Sim leaf must hold traj.tsv");
        assert!(
            rec.get("design_point_index").is_none(),
            "the legacy design marker must be gone; run.json = {rec}"
        );
    }

    // The experiment-side metadata is preserved exactly.
    let pts = output.join("designs/sens/parameter_points.tsv");
    assert!(pts.exists(), "parameter_points.tsv must still be written under designs/<name>/");
    let pts_txt = std::fs::read_to_string(&pts).unwrap();
    assert!(pts_txt.starts_with("point_id\tmu"), "parameter_points.tsv header: {pts_txt}");

    // No ad-hoc design sims tree.
    assert!(
        !output.join("designs/sens/sims").exists(),
        "the legacy designs/<name>/sims/ tree must not be written"
    );
}

/// (1) A partial/incomplete leaf is NOT served as a cache hit. After a clean
/// run, mark one leaf's `run.json` `running` (status != Completed → the CAS
/// `lookup` returns `Stale(Incomplete)`, never `Hit`). A re-run must re-execute
/// that cell and restore it to a valid `completed` leaf.
#[test]
fn incomplete_design_leaf_is_not_a_cache_hit() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("pd.camdl");
    let output = tmp.path().join("output");
    write_design_model(&model);
    let batch = tmp.path().join("batch.toml");
    write_design_batch(&batch, &model, &output, 2, &[1]);

    let first = run_batch(&bin, &batch);
    assert!(first.status.success(), "first run failed:\n{}", String::from_utf8_lossy(&first.stderr));

    let leaves = run_leaves(&output.join("sims"));
    assert_eq!(leaves.len(), 2, "expected 2 cells, got {:?}", leaves);

    // Corrupt one leaf into an INCOMPLETE state: flip its run.json status to
    // `running` (a crashed/in-flight leaf). The CAS lookup must NOT treat this
    // as a hit. We also drop traj.tsv so a stale read would be obvious.
    let victim = &leaves[0];
    let rj = victim.join("run.json");
    let mut rec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&rj).unwrap()).unwrap();
    rec["status"] = serde_json::Value::String("running".to_string());
    std::fs::write(&rj, serde_json::to_string_pretty(&rec).unwrap()).unwrap();
    let _ = std::fs::remove_file(victim.join("traj.tsv"));

    // Re-run (no --force). The incomplete cell must re-run; the other is a hit.
    let second = run_batch(&bin, &batch);
    assert!(
        second.status.success(),
        "second run failed:\n{}",
        String::from_utf8_lossy(&second.stderr)
    );

    // On-disk dedupe (precision-safe — same batch ⇒ same f64 point values): the
    // re-run resolves every cell to the SAME `Sim` run_id/path, so the
    // uncorrupted cell is a CAS hit (not re-created) and the victim re-fills its
    // own leaf in place. Leaf count stays 2 — no duplicate leaf is spawned.
    assert_eq!(
        run_leaves(&output.join("sims")).len(),
        2,
        "re-run must dedupe to the same leaves (CAS hit), not spawn duplicates"
    );

    // The victim leaf is valid + Completed again with its traj.tsv restored —
    // proof it re-ran rather than being skipped as a hit.
    let rec2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&rj).unwrap()).unwrap();
    assert_eq!(
        rec2["status"].as_str(),
        Some("completed"),
        "an incomplete leaf must re-run to Completed, not be served as a hit; run.json = {rec2}"
    );
    assert!(
        victim.join("traj.tsv").exists(),
        "the re-run must restore the cell's traj.tsv"
    );
}

/// (4) `--dry-run` reports cache hits via the real CAS identity/path (resolve +
/// `store.lookup`), not the legacy marker. A fresh output shows all misses; a
/// re-run after a real commit shows hits.
#[test]
fn design_dry_run_reports_cas_hits() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("pd.camdl");
    let output = tmp.path().join("output");
    write_design_model(&model);
    let batch = tmp.path().join("batch.toml");
    write_design_batch(&batch, &model, &output, 3, &[1]);

    // Dry-run on an empty store: 3 misses, 0 hits, no files written.
    let dry_fresh = Command::new(&bin)
        .args(["batch", "run", "--dry-run", &batch.to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(
        dry_fresh.status.success(),
        "design dry-run failed:\n{}",
        String::from_utf8_lossy(&dry_fresh.stderr)
    );
    let dry_fresh_err = String::from_utf8_lossy(&dry_fresh.stderr);
    assert!(
        dry_fresh_err.contains("3 cache misses"),
        "fresh dry-run must report 3 misses; stderr:\n{dry_fresh_err}"
    );
    assert!(
        dry_fresh_err.contains("0 cache hits"),
        "fresh dry-run must report 0 hits; stderr:\n{dry_fresh_err}"
    );
    // Dry-run writes NO sim leaves.
    assert!(
        run_leaves(&output.join("sims")).is_empty(),
        "dry-run must not commit any sim leaves"
    );

    // Real run commits the 3 cells.
    let run = run_batch(&bin, &batch);
    assert!(run.status.success(), "real run failed:\n{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(run_leaves(&output.join("sims")).len(), 3);

    // Dry-run again: now all 3 resolve to a Completed CAS leaf → 3 hits.
    let dry_warm = Command::new(&bin)
        .args(["batch", "run", "--dry-run", &batch.to_string_lossy()])
        .output()
        .expect("spawn");
    assert!(dry_warm.status.success(), "warm dry-run failed");
    let dry_warm_err = String::from_utf8_lossy(&dry_warm.stderr);
    assert!(
        dry_warm_err.contains("3 cache hits"),
        "warm dry-run must report 3 CAS hits (resolve + store.lookup); stderr:\n{dry_warm_err}"
    );
    assert!(
        dry_warm_err.contains("0 cache misses"),
        "warm dry-run must report 0 misses; stderr:\n{dry_warm_err}"
    );
}
