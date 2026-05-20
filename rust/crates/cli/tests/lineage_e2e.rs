//! End-to-end integration test for the three-layer lineage path (Layers 1–2).
//!
//! A `.camdl` model with `#[lineage]` → compile → `simulate --event-log` (TSV
//! and Parquet) → `lineage realize` (→ line list) → `lineage tree` → Newick →
//! assert tree properties.
//!
//! Silent-skip if the release `camdl` binary is not built or the colocated
//! `camdlc` version mismatches (same convention as `compile_output_flag.rs`).
//! Sets `CAMDL_SKIP_VERSION_CHECK=1` so a stale globally-installed `camdlc`
//! doesn't make the test flaky in a dev tree.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../target/release/camdl");
    p.exists().then_some(p)
}

fn tempdir(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "camdl_lineage_e2e_{}_{}_{}",
        tag,
        std::process::id(),
        ns
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

const SIR_LINEAGE: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
}
let N = S + I + R
transitions {
  #[lineage]
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
init { S = 499  I = 1 }
simulate { from = 0 'days  to = 60 'days }
"#;

fn run(camdl: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

/// Count leaf labels (`indN`) in a Newick string — a crude tip count.
fn count_tips(newick: &str) -> usize {
    // Tips are `indN` tokens; internal nodes are unlabelled. Count `ind`
    // occurrences (tip labels). Branch lengths use ':' so no false positives.
    newick.matches("ind").count()
}

#[test]
fn lineage_end_to_end_tsv_and_parquet() {
    let Some(camdl) = camdl_bin() else {
        eprintln!("skipping: release camdl binary not built");
        return;
    };

    let tmp = tempdir("e2e");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();

    // 1. Compile to IR JSON (skip whole test if camdlc unavailable/mismatched).
    let ir = tmp.join("sir.ir.json");
    let compiled = run(
        &camdl,
        &["compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
    );
    if !compiled.status.success() {
        eprintln!(
            "skipping: camdl compile failed (camdlc likely unavailable): {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        return;
    }

    let model_arg = ir.to_str().unwrap();
    let common = [
        "simulate",
        model_arg,
        "--backend",
        "gillespie",
        "--seed",
        "7",
        "--param",
        "beta=0.6",
        "--param",
        "gamma=0.2",
        "--param",
        "N0=500",
    ];

    // 2a. TSV event log (Layer 1) + emit the count trajectory.
    let ev_tsv = tmp.join("event_log.tsv");
    let traj_tsv = tmp.join("traj_lin.tsv");
    let ev_tsv_s = ev_tsv.to_str().unwrap();
    let traj_tsv_s = traj_tsv.to_str().unwrap();
    let mut args: Vec<&str> = common.to_vec();
    args.extend(["--event-log", ev_tsv_s, "--tsv", "--output", traj_tsv_s]);
    let out_tsv = run(&camdl, &args);
    assert!(
        out_tsv.status.success(),
        "simulate --event-log --tsv failed: {}",
        String::from_utf8_lossy(&out_tsv.stderr)
    );
    assert!(ev_tsv.exists(), "TSV event log not written");

    // 2b. Parquet event log.
    let ev_pq = tmp.join("event_log.parquet");
    let ev_pq_s = ev_pq.to_str().unwrap();
    let mut args2: Vec<&str> = common.to_vec();
    args2.extend(["--event-log", ev_pq_s, "--obs-only", "/dev/null"]);
    let out_pq = run(&camdl, &args2);
    assert!(
        out_pq.status.success(),
        "simulate --event-log (parquet) failed: {}",
        String::from_utf8_lossy(&out_pq.stderr)
    );
    assert!(ev_pq.exists(), "Parquet event log not written");

    // 3. Trajectory byte-identity (Tier 2a) at the CLI level: a run WITHOUT
    //    --event-log must produce the same trajectory bytes (the recorder draws
    //    no identities, so the simulation is literally unchanged).
    let traj_base = tmp.join("traj_base.tsv");
    let traj_base_s = traj_base.to_str().unwrap();
    let base = run(
        &camdl,
        &[
            "simulate", model_arg, "--backend", "gillespie", "--seed", "7",
            "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500",
            "--output", traj_base_s,
        ],
    );
    assert!(base.status.success(), "baseline simulate failed");
    let base_bytes = std::fs::read(&traj_base).unwrap();
    let lin_bytes = std::fs::read(&traj_tsv).unwrap();
    assert_eq!(
        base_bytes, lin_bytes,
        "CLI: count trajectory must be byte-identical with and without --event-log"
    );

    // 3b. Realize each event log (Layer 2) into a line list. TSV log →
    //     TSV line list; Parquet log → Parquet line list. Same identity seed.
    let ll_tsv = tmp.join("ll.tsv");
    let ll_tsv_s = ll_tsv.to_str().unwrap();
    let r1 = run(
        &camdl,
        &["lineage", "realize", ev_tsv_s, "--identity-seed", "7", "-o", ll_tsv_s],
    );
    assert!(
        r1.status.success(),
        "lineage realize (tsv) failed: {}",
        String::from_utf8_lossy(&r1.stderr)
    );
    assert!(ll_tsv.exists(), "TSV line list not written by realize");

    let ll_pq = tmp.join("ll.parquet");
    let ll_pq_s = ll_pq.to_str().unwrap();
    let r2 = run(
        &camdl,
        &["lineage", "realize", ev_pq_s, "--identity-seed", "7", "-o", ll_pq_s],
    );
    assert!(
        r2.status.success(),
        "lineage realize (parquet) failed: {}",
        String::from_utf8_lossy(&r2.stderr)
    );
    assert!(ll_pq.exists(), "Parquet line list not written by realize");

    // 4. Offline tree from TSV at flat:1.0 (all tips).
    let tree_tsv = tmp.join("tree_tsv.nwk");
    let tree_tsv_s = tree_tsv.to_str().unwrap();
    let t1 = run(
        &camdl,
        &[
            "lineage", "tree", ll_tsv_s, "--scheme", "flat:1.0", "--output",
            tree_tsv_s,
        ],
    );
    assert!(
        t1.status.success(),
        "lineage tree (tsv) failed: {}",
        String::from_utf8_lossy(&t1.stderr)
    );
    let nwk_tsv = std::fs::read_to_string(&tree_tsv).unwrap();
    assert!(nwk_tsv.trim_end().ends_with(';'), "Newick must end with ';'");
    let tips_tsv = count_tips(&nwk_tsv);
    assert!(tips_tsv > 0, "tree must have tips");

    // 5. Offline tree from Parquet at flat:1.0 → identical to TSV tree.
    let tree_pq = tmp.join("tree_pq.nwk");
    let tree_pq_s = tree_pq.to_str().unwrap();
    let t2 = run(
        &camdl,
        &[
            "lineage", "tree", ll_pq_s, "--scheme", "flat:1.0", "--output",
            tree_pq_s,
        ],
    );
    assert!(
        t2.status.success(),
        "lineage tree (parquet) failed: {}",
        String::from_utf8_lossy(&t2.stderr)
    );
    let nwk_pq = std::fs::read_to_string(&tree_pq).unwrap();
    assert_eq!(
        nwk_tsv, nwk_pq,
        "TSV and Parquet line lists must yield identical trees at flat:1.0"
    );

    // 6. Subsampled tree (flat:0.3) has fewer-or-equal tips than the full tree.
    let tree_sub = tmp.join("tree_sub.nwk");
    let tree_sub_s = tree_sub.to_str().unwrap();
    let t3 = run(
        &camdl,
        &[
            "lineage", "tree", ll_tsv_s, "--scheme", "flat:0.3", "--sample-seed", "3",
            "--output", tree_sub_s,
        ],
    );
    assert!(t3.status.success(), "lineage tree (subsample) failed");
    let nwk_sub = std::fs::read_to_string(&tree_sub).unwrap();
    assert!(
        count_tips(&nwk_sub) <= tips_tsv,
        "subsampled tree should have <= full tip count"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// ODE + --event-log must hard-error (capability gate), not silently produce
/// nothing.
#[test]
fn lineage_on_ode_is_rejected() {
    let Some(camdl) = camdl_bin() else { return };
    let tmp = tempdir("ode_reject");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();
    let ir = tmp.join("sir.ir.json");
    let compiled = run(
        &camdl,
        &["compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
    );
    if !compiled.status.success() {
        return; // camdlc unavailable
    }
    let out = run(
        &camdl,
        &[
            "simulate", ir.to_str().unwrap(), "--backend", "ode", "--seed", "1",
            "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500",
            "--event-log", tmp.join("x.tsv").to_str().unwrap(), "--tsv",
        ],
    );
    assert!(!out.status.success(), "ODE + --event-log must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ODE") || stderr.contains("incompatible"),
        "error should explain ODE incompatibility, got: {}",
        stderr
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
