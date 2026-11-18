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
    Command::new(camdl).env_remove("CLICOLOR_FORCE")
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

/// The leaf's run id and its manifest (`run.json`'s `artifacts`, serialized).
fn manifest_of(leaf: &Path) -> (String, String) {
    let rec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
    (rec["run_id"].as_str().unwrap().to_string(), rec["artifacts"].to_string())
}

/// `simulate --event-log` writes the event log as a child of the leaf at its
/// own address, `<leaf>/event_log/`, so what one run id holds does not depend
/// on whether any invocation asked for the log. A plain run, then the same
/// command with `--event-log`, then a plain run again leave the leaf's run id
/// and manifest byte-identical; the log is at the child address, declared in
/// `run.json`'s `children`, and is not one of the leaf's own files.
#[test]
fn the_event_log_is_a_child_and_the_leaf_manifest_never_changes() {
    let camdl = camdl_bin();
    let tmp = tempdir("evchild");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();

    let ir = tmp.join("sir.ir.json");
    let compiled = run(
        &camdl,
        &["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
    );
    if !compiled.status.success() {
        eprintln!("skipping: camdl compile failed (camdlc likely unavailable)");
        return;
    }

    let store = tmp.join("store");
    let store_s = store.to_str().unwrap();
    let model_arg = ir.to_str().unwrap();
    let base: Vec<&str> = vec![
        "simulate", model_arg, "--backend", "gillespie", "--seed", "7",
        "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500",
        "--output-dir", store_s,
    ];

    let first = run(&camdl, &base);
    assert!(first.status.success(), "plain simulate failed: {}",
        String::from_utf8_lossy(&first.stderr));
    let leaves = sim_leaves(&store);
    assert_eq!(leaves.len(), 1, "expected one sim leaf, got {leaves:?}");
    let leaf = leaves[0].clone();
    let plain = manifest_of(&leaf);
    let child = leaf.join("event_log").join("event_log.tsv");
    assert!(!child.exists(), "a plain run writes no event log");

    let ev = tmp.join("ev.tsv");
    let mut args = base.clone();
    args.extend(["--event-log", ev.to_str().unwrap()]);
    let second = run(&camdl, &args);
    assert!(second.status.success(), "simulate --event-log failed: {}",
        String::from_utf8_lossy(&second.stderr));
    assert_eq!(sim_leaves(&store), vec![leaf.clone()], "the same leaf, not a fork");
    assert_eq!(manifest_of(&leaf), plain,
        "--event-log leaves the leaf's run id and manifest byte-identical");
    assert!(!leaf.join("event_log.tsv").exists(), "the log is not one of the leaf's own files");
    assert!(child.is_file(), "the log is at the child address {}", child.display());
    assert!(leaf.join("event_log").join("event_log.json").is_file(), "the child carries its record");
    assert_eq!(std::fs::read(&ev).unwrap(), std::fs::read(&child).unwrap(),
        "the --event-log PATH mirror holds the child's bytes");
    let rec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
    assert!(rec["children"]["event_log"].as_array().is_some_and(|ids| ids.len() == 1),
        "run.json declares the child: {}", rec["children"]);

    // A later plain run reads the leaf as a valid hit (the child is declared,
    // so it is not an orphan of the leaf's exact set), and both survive it.
    let third = run(&camdl, &base);
    assert!(third.status.success(), "third run failed: {}",
        String::from_utf8_lossy(&third.stderr));
    assert_eq!(sim_leaves(&store), vec![leaf.clone()]);
    assert_eq!(manifest_of(&leaf), plain, "a later plain run leaves the manifest as it was");
    assert!(child.is_file(), "the child survives a later plain run");
}

/// A compiled lineage model and a store in a fresh directory, with the
/// `simulate` arguments every case below shares. `None` when camdlc is
/// unavailable (the test then skips, as the rest of this file does).
struct Lineage {
    camdl: PathBuf,
    tmp: PathBuf,
    ir: String,
    store: String,
}

impl Lineage {
    fn new(tag: &str) -> Option<Lineage> {
        let camdl = camdl_bin();
        let tmp = tempdir(tag);
        let model = tmp.join("sir.camdl");
        std::fs::write(&model, SIR_LINEAGE).unwrap();
        let ir = tmp.join("sir.ir.json");
        let compiled = run(&camdl, &["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()]);
        if !compiled.status.success() {
            eprintln!("skipping: camdl compile failed (camdlc likely unavailable)");
            return None;
        }
        let store = tmp.join("store").to_string_lossy().into_owned();
        Some(Lineage { camdl, tmp, ir: ir.to_string_lossy().into_owned(), store })
    }

    /// `camdl simulate` on the model into the store, with `extra`.
    fn simulate(&self, extra: &[&str]) -> std::process::Output {
        let mut args: Vec<&str> = vec![
            "simulate", &self.ir, "--backend", "gillespie", "--seed", "7",
            "--param", "beta=0.6", "--param", "gamma=0.2", "--param", "N0=500",
            "--output-dir", &self.store,
        ];
        args.extend(extra);
        run(&self.camdl, &args)
    }

    /// `camdl cat <leaf> --stream event_log.tsv`.
    fn cat_event_log(&self, leaf: &Path) -> std::process::Output {
        run(&self.camdl, &["cat", leaf.to_str().unwrap(), "--root", &self.store, "--stream", "event_log.tsv"])
    }

    /// The store's one sim leaf.
    fn leaf(&self) -> PathBuf {
        let leaves = sim_leaves(Path::new(&self.store));
        assert_eq!(leaves.len(), 1, "expected one sim leaf, got {leaves:?}");
        leaves[0].clone()
    }
}

fn ok(o: &std::process::Output, what: &str) {
    assert!(o.status.success(), "{what} failed: {}", String::from_utf8_lossy(&o.stderr));
}

fn record(leaf: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap()
}

/// The id a leaf declares for its event-log child is SHA-256 of
/// `"{run_id}:event_log"`, hex: pinned, because a reader that finds the child
/// through `children` and the writer that names it must agree.
#[test]
fn the_event_log_child_id_is_pinned() {
    use sha2::{Digest, Sha256};
    let Some(fx) = Lineage::new("evid") else { return };
    ok(&fx.simulate(&["--event-log"]), "simulate --event-log");
    let rec = record(&fx.leaf());
    let run_id = rec["run_id"].as_str().unwrap();
    let want: String = Sha256::digest(format!("{run_id}:event_log").as_bytes())
        .iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(rec["children"]["event_log"], serde_json::json!([want]));
}

/// `--stdout` takes the run out of the store, so there is no leaf for the
/// event log to be a child of, and the log would be dropped with exit 0. The
/// combination is refused.
#[test]
fn stdout_and_event_log_are_refused_together() {
    let Some(fx) = Lineage::new("evstdout") else { return };
    let ev = fx.tmp.join("ev.tsv");
    for extra in [vec!["--stdout", "--event-log"], vec!["--stdout", "--event-log", ev.to_str().unwrap()]] {
        let o = fx.simulate(&extra);
        assert!(!o.status.success(), "{extra:?} must be refused, not drop the log");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("--stdout") && err.contains("--event-log"), "the refusal names both flags: {err}");
    }
    assert!(!ev.exists());
}

/// `camdl cat --stream event_log.tsv` reads the child only when it is
/// complete: its record present and its file the size the record names. A
/// child without its record, or whose file differs from it, is a clear error
/// that names the remedy, never a partial log on stdout.
#[test]
fn cat_reads_only_a_complete_event_log_child() {
    let Some(fx) = Lineage::new("evcat") else { return };
    ok(&fx.simulate(&["--event-log"]), "simulate --event-log");
    let leaf = fx.leaf();
    let file = leaf.join("event_log").join("event_log.tsv");
    let rec = leaf.join("event_log").join("event_log.json");
    let full = std::fs::read(&file).unwrap();
    let o = fx.cat_event_log(&leaf);
    ok(&o, "cat --stream event_log.tsv");
    assert_eq!(o.stdout, full, "cat prints the child's bytes");

    let refused = |what: &str| {
        let o = fx.cat_event_log(&leaf);
        assert!(!o.status.success(), "{what}: cat must refuse");
        assert!(o.stdout.is_empty(), "{what}: nothing on stdout");
        let err = String::from_utf8_lossy(&o.stderr);
        assert!(err.contains("--event-log") && err.contains("--force"), "{what}: names the remedy: {err}");
    };
    std::fs::write(&file, &full[..full.len() / 2]).unwrap();
    refused("a file shorter than its record");
    std::fs::write(&file, &full).unwrap();
    std::fs::remove_file(&rec).unwrap();
    refused("a file without its record");
}

/// The crash state — the child declared in `run.json` but nothing written —
/// is readable as "not there": `cat` refuses cleanly, a plain rerun reads the
/// leaf as a hit (its record, the declaration included, is kept), and an
/// `--event-log` rerun writes the child.
#[test]
fn a_declared_but_empty_event_log_child_recovers() {
    use sha2::{Digest, Sha256};
    let Some(fx) = Lineage::new("evcrash") else { return };
    ok(&fx.simulate(&[]), "plain simulate");
    let leaf = fx.leaf();
    let mut rec = record(&leaf);
    let run_id = rec["run_id"].as_str().unwrap().to_string();
    let id: String = Sha256::digest(format!("{run_id}:event_log").as_bytes())
        .iter().map(|b| format!("{b:02x}")).collect();
    rec["children"] = serde_json::json!({ "event_log": [id] });
    std::fs::write(leaf.join("run.json"), serde_json::to_vec_pretty(&rec).unwrap()).unwrap();
    let declared = std::fs::read(leaf.join("run.json")).unwrap();

    let o = fx.cat_event_log(&leaf);
    assert!(!o.status.success() && o.stdout.is_empty(), "cat refuses a declared, empty child");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("--event-log"), "the refusal names the remedy: {err}");

    ok(&fx.simulate(&[]), "plain rerun");
    assert_eq!(fx.leaf(), leaf);
    assert_eq!(std::fs::read(leaf.join("run.json")).unwrap(), declared,
        "a plain rerun is a hit: the record, its declaration included, is kept as it was");

    ok(&fx.simulate(&["--event-log"]), "simulate --event-log");
    let o = fx.cat_event_log(&leaf);
    ok(&o, "cat after the --event-log rerun");
    assert!(o.stdout.starts_with(b"# camdl-event-log"), "the child is written");
}

/// Committed sim leaves (dirs holding a run.json) under `<store>/sims`.
fn sim_leaves(store: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![store.join("sims")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").exists() { out.push(p); } else { stack.push(p); }
            }
        }
    }
    out
}

#[test]
fn lineage_end_to_end_tsv_and_parquet() {
    let camdl = camdl_bin();

    let tmp = tempdir("e2e");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();

    // 1. Compile to IR JSON (skip whole test if camdlc unavailable/mismatched).
    let ir = tmp.join("sir.ir.json");
    let compiled = run(
        &camdl,
        &["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
    );
    if !compiled.status.success() {
        eprintln!(
            "skipping: camdl compile failed (camdlc likely unavailable): {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        return;
    }

    let model_arg = ir.to_str().unwrap();
    let store = tmp.join("store");
    let store_s = store.to_str().unwrap();
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
        "--output-dir",
        store_s,
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
    // No `-o`: the trajectory goes to the CAS leaf only (the event log is the
    // artifact of interest here). The Parquet event log is mirrored to ev_pq.
    let mut args2: Vec<&str> = common.to_vec();
    args2.extend(["--event-log", ev_pq_s]);
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
            "--output-dir", store_s, "--output", traj_base_s,
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

    // 3c. The offline projections (tree/sojourn/cohort) are NOT
    //     content-addressed, so `--output PATH` is the only sink: invoked
    //     without it they MUST exit non-zero with a clear error and write
    //     NOTHING to stdout (no Newick/TSV stream dumped to a pipe).
    for (sub, extra) in [
        ("tree", vec!["--scheme", "flat:1.0"]),
        ("sojourn", vec!["--compartment", "1"]),
        ("cohort", vec!["--event", "infection", "--window", "7"]),
    ] {
        let mut args: Vec<&str> = vec!["lineage", sub, ll_tsv_s];
        args.extend(extra);
        let out = run(&camdl, &args);
        assert!(
            !out.status.success(),
            "lineage {sub} without --output must exit non-zero"
        );
        assert!(
            out.stdout.is_empty(),
            "lineage {sub} without --output must write NOTHING to stdout, got {} bytes",
            out.stdout.len()
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("requires --output"),
            "lineage {sub} without --output must explain the requirement, got: {stderr}"
        );
    }

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
    let camdl = camdl_bin();
    let tmp = tempdir("ode_reject");
    let model = tmp.join("sir.camdl");
    std::fs::write(&model, SIR_LINEAGE).unwrap();
    let ir = tmp.join("sir.ir.json");
    let compiled = run(
        &camdl,
        &["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
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

/// A `#[lineage]` competing exit from a compartment staged by `via erlang`,
/// with a real parent inside a rate the compiler lowers to the guarded share
/// `if E > 0 then R * E_s / E else 0` (gh#1276). The parent-pool weight of
/// each stage copy is a conditional; the run records lineage events for the
/// copies, each with one finite, non-negative parent mass.
const STAGED_EXIT_LINEAGE: &str = r#"
time_unit = 'days
dimensions { age = [pre, sch] }
compartments { S, E, I, R, T }
stratify(by = age)
parameters { beta : rate  sigma : rate  gamma : rate  mu : rate  K : count }
transitions {
  infection[a in age] : S[a] --> E[a] @ beta * S[a] * I / (S + E + I + R)
  latency[a in age]   : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
  recovery[a in age]  : I[a] --> R[a] @ gamma * I[a]
  #[lineage]
  treat[a in age] : E[a] --> T[a] @ mu * E[a] * I[a] / (K + E[a])
}
init { S[pre] = 500  S[sch] = 490  E[pre] = 10  I[pre] = 5 }
simulate { from = 0 'days  to = 30 'days }
"#;

#[test]
fn staged_competing_exit_records_conditional_parent_weights() {
    let camdl = camdl_bin();
    let tmp = tempdir("staged_exit");
    let model = tmp.join("staged.camdl");
    std::fs::write(&model, STAGED_EXIT_LINEAGE).unwrap();
    let ir = tmp.join("staged.ir.json");
    let compiled = run(
        &camdl,
        &["dev", "compile", model.to_str().unwrap(), "-o", ir.to_str().unwrap()],
    );
    if !compiled.status.success() {
        let stderr = String::from_utf8_lossy(&compiled.stderr);
        // A compiler diagnostic is this test's failure; anything else means
        // camdlc is unavailable, as for the other tests here.
        assert!(!stderr.contains("[E"), "the model must compile: {stderr}");
        eprintln!("skipping: camdl compile failed (camdlc likely unavailable): {stderr}");
        return;
    }

    // The stage copies of `treat_pre`, by their index in the compiled model.
    let compiled_ir: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ir).unwrap()).unwrap();
    let copies: Vec<usize> = compiled_ir["model"]["transitions"]
        .as_array()
        .expect("the compiled model lists its transitions")
        .iter()
        .enumerate()
        .filter(|(_, t)| t["name"].as_str().is_some_and(|n| n.starts_with("treat_pre_s")))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(copies.len(), 3, "treat_pre has one copy per stage");

    let ev = tmp.join("event_log.tsv");
    let store = tmp.join("store");
    let out = run(
        &camdl,
        &[
            "simulate", ir.to_str().unwrap(), "--backend", "gillespie", "--seed", "7",
            "--param", "beta=0.4", "--param", "sigma=0.2", "--param", "gamma=0.1",
            "--param", "mu=0.5", "--param", "K=5",
            "--output-dir", store.to_str().unwrap(),
            "--event-log", ev.to_str().unwrap(), "--tsv",
        ],
    );
    assert!(out.status.success(), "simulate --event-log failed: {}",
        String::from_utf8_lossy(&out.stderr));

    // Columns: time, transition, multiplicity, batched, step, lineage_weights.
    let text = std::fs::read_to_string(&ev).unwrap();
    let mut treat_events = 0;
    for row in text.lines().skip(4) {
        let f: Vec<&str> = row.split('\t').collect();
        let tr: usize = f[1].parse().unwrap();
        if !copies.contains(&tr) {
            continue;
        }
        treat_events += 1;
        let masses: Vec<f64> = serde_json::from_str(f[5])
            .unwrap_or_else(|e| panic!("treat event weights {:?}: {e}", f[5]));
        assert_eq!(masses.len(), 1, "one parent pool, I_pre: {row}");
        assert!(masses[0].is_finite() && masses[0] >= 0.0, "a finite mass: {row}");
    }
    assert!(treat_events > 0, "the run must fire a stage copy of treat_pre");

    let ll = tmp.join("ll.tsv");
    let realized = run(
        &camdl,
        &["lineage", "realize", ev.to_str().unwrap(), "--identity-seed", "7",
          "-o", ll.to_str().unwrap()],
    );
    assert!(realized.status.success(), "lineage realize failed: {}",
        String::from_utf8_lossy(&realized.stderr));

    let _ = std::fs::remove_dir_all(&tmp);
}
