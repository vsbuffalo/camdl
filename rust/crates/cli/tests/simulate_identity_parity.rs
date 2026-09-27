//! Run-identity parity between simulate's run path and its identity path
//! (audit 2026-08-23 #1/#2; proposal
//! docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md Phase 1).
//!
//! `resolve_run_model` applies `--integrator` and `--param-vec` to the model
//! that is RUN; `build_simulate_cas_sink` builds the model that is HASHED in
//! a second load. Before the fix the second load skipped both overrides, so
//! two runs with different settings shared one `run_id` — the second was
//! silently served the first's trajectory (pre-S1), or died with
//! DivergentRecompute (post-S1). These tests pin that each override splits
//! the leaves: two settings, two CAS leaves, both runs succeeding.
//!
//! Shells out to the built `camdl` binary; skipped silently when the release
//! binary or `camdlc.exe` isn't present (rust-only CI / pre-build).

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../target/release/camdl")
}
fn camdlc() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct TempDir(PathBuf);
impl TempDir { fn path(&self) -> &Path { &self.0 } }
impl Drop for TempDir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir()
        .join(format!("camdl_idparity_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

fn compile(dir: &Path, camdlc: &Path, src: &str, stem: &str) -> PathBuf {
    let model_path = dir.join(format!("{stem}.camdl"));
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join(format!("{stem}.ir.json"));
    let out = Command::new(camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();
    ir_path
}

/// Count committed sim leaves (dirs holding a run.json) under `<out>/sims`.
fn sim_leaves(out: &Path) -> Vec<PathBuf> {
    let mut leaves = Vec::new();
    let mut stack = vec![out.join("sims")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").exists() {
                    leaves.push(p);
                } else {
                    stack.push(p);
                }
            }
        }
    }
    leaves
}

const SIR_ODE: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 60 'days }
"#;

#[test]
fn integrator_override_splits_the_cas_leaves() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("integrator");
    let ir = compile(tmp.path(), &cc, SIR_ODE, "sir");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.5\ngamma = 0.25\nN0 = 10000\n").unwrap();
    let out = tmp.path().join("out");

    for m in ["rk4", "rk45"] {
        let st = Command::new(&bin)
            .args(["simulate"]).arg(&ir)
            .args(["--params"]).arg(&params)
            .args(["--backend", "ode", "--dt", "1", "--seed", "1",
                   "--integrator", m,
                   "--cas", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("traj_{m}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().unwrap();
        assert!(st.status.success(),
            "simulate --integrator {m} must succeed (a DivergentRecompute here \
             means the override is not in the run identity): {}",
            String::from_utf8_lossy(&st.stderr));
    }
    assert_eq!(sim_leaves(&out).len(), 2,
        "rk4 and rk45 runs must land in two distinct CAS leaves");
}

/// `--force` must actually replace the stored artifact. It never did: batch
/// recomputed the cell and the fresh bytes were discarded at commit as an
/// already-completed no-op, so the leaf kept its original content. The store
/// now has an overwrite door (displace-then-recompute, incumbent quarantined).
#[test]
fn force_replaces_the_stored_leaf() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("force");
    let ir = compile(tmp.path(), &cc, SIR_ODE, "sir");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.5\ngamma = 0.25\nN0 = 10000\n").unwrap();
    let out = tmp.path().join("out");

    let run = |extra: &[&str]| {
        let mut cmd = Command::new(&bin);
        cmd.args(["simulate"]).arg(&ir)
            .args(["--params"]).arg(&params)
            .args(["--backend", "ode", "--dt", "1", "--seed", "1",
                   "--cas", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join("traj.tsv"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1");
        for a in extra { cmd.arg(a); }
        let o = cmd.output().unwrap();
        assert!(o.status.success(), "simulate failed: {}", String::from_utf8_lossy(&o.stderr));
    };

    // The observable is the quarantine: forcing DISPLACES the incumbent (and
    // preserves it) before recomputing. Content is not an observable here —
    // the rebuild is deterministic, so the bytes are identical either way;
    // and tampering with the leaf would prove nothing, since the exact-set
    // check (size + mtime) already reclaims a modified leaf without --force.
    let quarantine = out.join(".quarantine");
    let q_count = || -> usize {
        std::fs::read_dir(&quarantine).map(|d| d.count()).unwrap_or(0)
    };

    run(&[]);
    assert_eq!(sim_leaves(&out).len(), 1, "one leaf after the first run");
    let original = std::fs::read(sim_leaves(&out)[0].join("traj.tsv")).unwrap();

    // Negative control: a plain rerun is a cache hit and displaces nothing.
    run(&[]);
    assert_eq!(q_count(), 0, "a cache-hit rerun must not quarantine anything");

    run(&["--force"]);
    assert!(q_count() >= 1,
        "--force must displace the incumbent into .quarantine; without the \
         overwrite door it was a cache hit (or a discarded recompute) and \
         nothing was ever replaced");
    assert_eq!(sim_leaves(&out).len(), 1, "the forced rerun leaves one live leaf");
    assert_eq!(std::fs::read(sim_leaves(&out)[0].join("traj.tsv")).unwrap(), original,
        "the forced recompute reproduces the same deterministic trajectory");
}

const SIR_VEC: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta_a : rate  in [0.05, 5.0]
  beta_b : rate  in [0.05, 5.0]
  gamma  : rate  in [0.01, 1.0]
  N0     : count in [100, 100000]
}
transitions {
  infection_a : S --> I @ beta_a * S * I / N0
  infection_b : S --> I @ beta_b * S * I / N0
  recovery    : I --> R @ gamma * I
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 30 'days }
"#;

#[test]
fn param_vec_values_split_the_cas_leaves() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("paramvec");
    let ir = compile(tmp.path(), &cc, SIR_VEC, "sirvec");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "gamma = 0.25\nN0 = 10000\n").unwrap();
    let out = tmp.path().join("out");

    for (tag, a, b) in [("v1", 0.4, 0.6), ("v2", 0.9, 1.1)] {
        let vec_file = tmp.path().join(format!("beta_{tag}.tsv"));
        std::fs::write(&vec_file, format!("a\t{a}\nb\t{b}\n")).unwrap();
        let st = Command::new(&bin)
            .args(["simulate"]).arg(&ir)
            .args(["--params"]).arg(&params)
            .args(["--seed", "1",
                   "--param-vec"]).arg(format!("beta={}", vec_file.display()))
            .args(["--cas", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("traj_{tag}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .output().unwrap();
        assert!(st.status.success(),
            "simulate --param-vec ({tag}) must succeed (a DivergentRecompute \
             here means the vec values are not in the run identity): {}",
            String::from_utf8_lossy(&st.stderr));
    }
    assert_eq!(sim_leaves(&out).len(), 2,
        "two different --param-vec files must land in two distinct CAS leaves");
}

/// gh#583 (item C, the params half). The identity rebuilt a cell's parameter
/// map in a walk of its own — `--params`, `--param-vec`, `--param`, then the
/// draw row layered on top — while the engine resolves them through the
/// resolver, where a draw/sweep point sits BELOW `--param`. So a draws file
/// with a `beta` column, run once plain and once with `--param beta=…`, hashed
/// both to one `run_id` (the draw's β won in the identity) while simulating
/// two different β (the `--param` won in the run): the second run died with
/// DivergentRecompute, and before S1 it would have been served the first's
/// trajectory. Identity now hashes the values the resolver hands the engine.
#[test]
fn param_that_shadows_a_draw_splits_the_cas_leaves() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("drawshadow");
    let ir = compile(tmp.path(), &cc, SIR_ODE, "sir");
    let draws = tmp.path().join("d.tsv");
    std::fs::write(&draws, "beta\tgamma\tN0\n0.3\t0.1\t10000\n").unwrap();
    let out = tmp.path().join("out");

    for (tag, extra) in [("plain", None), ("shadowed", Some("beta=0.6"))] {
        let mut cmd = Command::new(&bin);
        cmd.args(["simulate"]).arg(&ir)
            .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--draws"]).arg(&draws)
            .args(["--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("traj_{tag}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1");
        if let Some(p) = extra { cmd.args(["--param", p]); }
        let st = cmd.output().unwrap();
        assert!(st.status.success(),
            "simulate --draws ({tag}) must succeed (a DivergentRecompute here \
             means the identity hashed a parameter value the run did not use): {}",
            String::from_utf8_lossy(&st.stderr));
    }
    let leaves = sim_leaves(&out);
    assert_eq!(leaves.len(), 2,
        "a draw and the same draw under `--param beta=0.6` simulate different β, \
         so they must land in two distinct CAS leaves");
    // The params path label names the values the cell ran: the shadowed leaf
    // must say β = 0.6, not the draw's 0.3.
    let labels: Vec<String> = leaves.iter()
        .map(|l| l.to_string_lossy().into_owned()).collect();
    assert!(labels.iter().any(|l| l.contains("beta_0.6")),
        "the shadowed leaf's params label must carry the β it ran: {labels:?}");
}

const SIR_DATED: &str = r#"
time_unit = 'days
origin = date("2020-01-01")
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
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

/// Every committed `SimEnsemble` leaf under `<out>/ensembles`.
fn ensemble_leaves(out: &Path) -> Vec<PathBuf> {
    let mut leaves = Vec::new();
    let mut stack = vec![out.join("ensembles")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").exists() { leaves.push(p) } else { stack.push(p) }
            }
        }
    }
    leaves
}

/// gh#583 item B. `--dates` adds `date` columns to the combined TSV, and the
/// multi-cell ensemble was stored from that same buffer — but `--dates` is in
/// none of the ensemble's identity levels. So whichever run landed first
/// decided the stored bytes: a `--dates` run stored a dated ensemble that a
/// plain run was then served, and the other order died in DivergentRecompute
/// (a warning, and no ensemble). `--dates` is presentation: the ensemble is
/// stored date-free, like the `Sim` leaves, and only the `-o` mirror is dated.
#[test]
fn dates_reach_the_mirror_and_not_the_stored_ensemble() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("dates");
    let ir = compile(tmp.path(), &cc, SIR_DATED, "sird");
    let out = tmp.path().join("out");
    let run = |tag: &str, dates: bool| {
        let mut cmd = Command::new(&bin);
        cmd.args(["simulate"]).arg(&ir)
            .args(["--param", "beta=0.3", "--param", "gamma=0.1", "--param", "N0=1000",
                   "--seeds", "1,2", "--output-dir"]).arg(&out)
            .arg("-o").arg(tmp.path().join(format!("{tag}.tsv")))
            .env("CAMDL_SKIP_VERSION_CHECK", "1");
        if dates { cmd.arg("--dates"); }
        let o = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(o.status.success(), "simulate ({tag}): {stderr}");
        stderr
    };
    let header = |p: &Path| -> String {
        std::fs::read_to_string(p).unwrap().lines().nth(1).unwrap_or("").to_string()
    };

    // `--dates` first: the run that decides the stored bytes.
    run("dated", true);
    assert!(header(&tmp.path().join("dated.tsv")).split('\t').any(|c| c == "date"),
        "the -o mirror of a --dates run carries the date column");
    let ens = ensemble_leaves(&out);
    assert_eq!(ens.len(), 1, "one ensemble leaf: {ens:?}");
    let stored = header(&ens[0].join("ensemble.tsv"));
    assert!(!stored.split('\t').any(|c| c == "date"),
        "the stored ensemble must be date-free — `--dates` is in no identity \
         level, so a plain run would be served these bytes: {stored}");

    // The plain run resolves the same ensemble identity and agrees with it.
    let stderr = run("plain", false);
    assert!(!stderr.contains("divergent recompute"),
        "a plain rerun must agree with the stored ensemble: {stderr}");
    assert_eq!(std::fs::read(tmp.path().join("plain.tsv")).unwrap(),
               std::fs::read(ens[0].join("ensemble.tsv")).unwrap(),
        "without --dates, the -o mirror and the stored ensemble are the same bytes");
}

const SIR_DEFAULTED: &str = r#"
time_unit = 'days
compartments { S, I, R }
let gamma : rate = 0.1
parameters {
  beta : rate  in [0.001, 5.0]
  N0   : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

/// gh#583 (the batch half). `batch run` hashed its params file's raw entries
/// as the cell's parameters, `simulate` hashed the model defaults overlaid by
/// the same file — two walks, neither the resolver's. On a model with a
/// defaulted parameter the same cell (same model, params file, seed) got two
/// `params` levels, so `simulate` and `batch run` each re-simulated what the
/// other had stored. Both now hash the resolved values of the cell's own
/// `SimRun`, so they meet on one leaf.
#[test]
fn simulate_and_batch_share_a_leaf_over_a_defaulted_parameter() {
    let bin = camdl_bin();
    let Some(cc) = camdlc() else {
        eprintln!("skip: camdlc.exe missing (run `make build`)");
        return;
    };
    if !bin.exists() {
        eprintln!("skip: release camdl missing (run `make build`)");
        return;
    }
    let tmp = tempdir("batchdefault");
    let ir = compile(tmp.path(), &cc, SIR_DEFAULTED, "m");
    let params = tmp.path().join("p.toml");
    std::fs::write(&params, "beta = 0.3\nN0 = 1000\n").unwrap();
    let out = tmp.path().join("store");

    let st = Command::new(&bin)
        .args(["simulate"]).arg(&ir)
        .args(["--params"]).arg(&params)
        .args(["--seed", "3", "--output-dir"]).arg(&out)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(st.status.success(), "simulate: {}", String::from_utf8_lossy(&st.stderr));

    let manifest = tmp.path().join("b.toml");
    std::fs::write(&manifest, format!(
        "[config]\nmodel = \"{}\"\nparams = \"{}\"\noutput_dir = \"{}\"\nseeds = {{ list = [3] }}\n",
        ir.display(), params.display(), out.display(),
    )).unwrap();
    let st = Command::new(&bin)
        .args(["batch", "run"]).arg(&manifest)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output().unwrap();
    assert!(st.status.success(), "batch run: {}", String::from_utf8_lossy(&st.stderr));

    let leaves = sim_leaves(&out);
    assert_eq!(leaves.len(), 1,
        "simulate and batch run of one cell must resolve to one leaf; got {leaves:?}");
}
