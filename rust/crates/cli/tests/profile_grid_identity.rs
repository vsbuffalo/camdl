//! A profile point's identity is its own values, not its place in a grid: the
//! order of the `--sweep` flags and of a listed grid's values changes neither
//! a point's run id nor what it stores, and the method's perturbation steps
//! are part of every point's identity.

use std::collections::BTreeSet;
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

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} - run `make build-ocaml`", p.display());
    p
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_profile_grid_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

fn write_fixture(dir: &Path) -> PathBuf {
    let camdlc = camdlc_bin();
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
    projected  = I
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).env_remove("CLICOLOR_FORCE").arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();
    // The base point the estimated parameters start from. A scenario that
    // set them would be refused: a parameter is estimated or given a value.
    std::fs::write(dir.join("start.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 1000\n").unwrap();

    // The fit problem the profiles are of: the two swept parameters and `N0`
    // are all estimated.
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"sir.ir.json\"\n\n[data.observations]\ncases = \"cases.tsv\"\n\n\
         [estimate]\nbeta = { bounds = [0.001, 5.0] }\ngamma = { bounds = [0.01, 1.0] }\n\
         N0 = { bounds = [100.0, 10000.0] }\n\n[fixed]\n").unwrap();
    ir_path
}

/// `--fit` naming the fixture's fit problem, which sits beside its model.
fn fit_arg(ir: &Path) -> String {
    ir.with_file_name("fit.toml").to_string_lossy().into_owned()
}

/// `--starts from_params=<start.toml>` beside the fixture's model.
fn starts_arg(ir: &Path) -> String {
    format!("from_params={}", ir.with_file_name("start.toml").to_string_lossy())
}

/// Run a profile with the given sweep specs and `--rw-sd` into the run store
/// at `root`.
fn run_profile_with(
    bin: &Path, ir: &Path, root: &Path,
    beta: &str, gamma: &str, rw_sd: &str,
) {
    let out = Command::new(bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--starts", &starts_arg(ir),
            "--fit", &fit_arg(ir),
            "--sweep", beta,
            "--sweep", gamma,
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "5",
            "--n-starts", "1",
            "--rw-sd", rw_sd,
            "--seed", "1",
        ])
        .output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile must exit 0.\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// Every profile point in the run store at `root`: its run id (from its
/// `identity.json`) and the bytes of its `mle.toml`.
fn leaves(root: &Path) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') { continue }
            let ident = p.join("identity.json");
            if ident.is_file() {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&ident).unwrap()).unwrap();
                if v["kind"] == "profile_point" {
                    let id = v["run_id"].as_str().expect("identity.json run_id").to_string();
                    out.insert(id, std::fs::read_to_string(p.join("mle.toml")).expect("a point's mle.toml"));
                    continue;
                }
            }
            stack.push(p);
        }
    }
    out
}

/// The perturbation magnitudes are part of the method, so they belong in the
/// key. `--rw-sd` is a REQUIRED flag whose values drive the IF2 sampler, yet
/// only the estimated *set* was keyed (incidentally, via the priors blob) —
/// the magnitudes were not. So `--rw-sd "beta=0.05"` after `--rw-sd
/// "beta=0.5"` was served the FIRST run's profile, cell for cell, under a
/// "cached — resuming" line (2026-08-23 audit).
#[test]
fn distinct_rw_sd_gets_distinct_points() {
    let bin = camdl_bin();
    let tmp = tempdir("rwsd");
    let ir = write_fixture(tmp.path());
    let root = tmp.path().join("out");
    let (beta, gamma) = ("beta=lin(0.1,0.5,3)", "gamma=lin(0.03,0.25,3)");

    // The magnitude must be on a non-focal parameter: beta and gamma are
    // pinned by the sweep, so only N0 is estimated here.
    run_profile_with(&bin, &ir, &root, beta, gamma, "N0=5.0");
    let after_a: BTreeSet<String> = leaves(&root).into_keys().collect();
    assert_eq!(after_a.len(), 9, "3 × 3 points, one start each, got {:?}", after_a);

    // The same magnitudes again find every point stored (the negative
    // control: this must not add a point, or the assertion below proves
    // nothing).
    run_profile_with(&bin, &ir, &root, beta, gamma, "N0=5.0");
    assert_eq!(leaves(&root).into_keys().collect::<BTreeSet<_>>(), after_a,
        "re-running with the same rw_sd must reuse its points");

    // Different magnitudes drive a different sampler: nine other points.
    run_profile_with(&bin, &ir, &root, beta, gamma, "N0=50.0");
    let after_b: BTreeSet<String> = leaves(&root).into_keys().collect();
    assert_eq!(after_b.len(), 18,
        "a different --rw-sd must store points of its own rather than being served \
         the previous run's landscape — got {:?}", after_b);
    assert!(after_b.is_superset(&after_a),
        "the first run's points must still be present alongside the second's");
}

/// Run a two-start profile with the sweep specs in the given order into `root`.
fn run_profile_sweeps(bin: &Path, ir: &Path, root: &Path, sweeps: &[&str]) {
    let mut cmd = Command::new(bin);
    cmd.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", root)
        .args([
            "profile", &ir.to_string_lossy(),
            "--starts", &starts_arg(ir),
            "--fit", &fit_arg(ir),
        ]);
    for s in sweeps {
        cmd.args(["--sweep", s]);
    }
    let out = cmd
        .args([
            "--algorithm", "if2",
            "--particles", "30",
            "--iterations", "3",
            "--n-starts", "2",
            "--rw-sd", "N0=50.0",
            "--seed", "1",
        ])
        .output().expect("spawn camdl profile");
    assert!(out.status.success(),
        "camdl profile must exit 0.\nstderr:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// A profile point's identity holds its focal values by name, so two commands
/// that differ only in the order of their `--sweep` flags name the same
/// points. They must then store the same result in each: a point's random
/// stream may not depend on where the flag order puts it in the grid. The
/// axes have different lengths (3 × 2), so swapping them moves the interior
/// points to different positions in the grid.
#[test]
fn sweep_flag_order_changes_neither_run_ids_nor_results() {
    let bin = camdl_bin();
    let tmp = tempdir("order");
    let ir = write_fixture(tmp.path());
    let (beta, gamma) = ("beta=lin(0.1,0.5,3)", "gamma=lin(0.05,0.25,2)");

    let root_a = tmp.path().join("out_a");
    let root_b = tmp.path().join("out_b");
    run_profile_sweeps(&bin, &ir, &root_a, &[beta, gamma]);
    run_profile_sweeps(&bin, &ir, &root_b, &[gamma, beta]);
    let a = leaves(&root_a);
    let b = leaves(&root_b);

    // The two starts begin at the start file's values, so the second is a
    // replicate of the first: two points per cell.
    assert_eq!(a.len(), 12, "3 × 2 cells × 2 starts, got {}", a.len());
    let ids_a: BTreeSet<&String> = a.keys().collect();
    let ids_b: BTreeSet<&String> = b.keys().collect();
    assert_eq!(ids_a, ids_b, "swapping the --sweep flags must not change any run id");
    for (id, mle_a) in &a {
        assert_eq!(mle_a, &b[id],
            "point {id} stores a different mle.toml when the --sweep flags are swapped");
    }
}

/// `beta=0.1,0.5` and `beta=0.5,0.1` name the same points. Each must hold the
/// same result under both, or a store holds whichever order ran first under
/// that run id.
#[test]
fn listed_value_order_changes_neither_run_ids_nor_results() {
    let bin = camdl_bin();
    let tmp = tempdir("valorder");
    let ir = write_fixture(tmp.path());
    let gamma = "gamma=0.05,0.25";

    let root_a = tmp.path().join("out_a");
    let root_b = tmp.path().join("out_b");
    run_profile_sweeps(&bin, &ir, &root_a, &["beta=0.1,0.3,0.5", gamma]);
    run_profile_sweeps(&bin, &ir, &root_b, &["beta=0.5,0.1,0.3", gamma]);
    let a = leaves(&root_a);
    let b = leaves(&root_b);

    // The two starts begin at the start file's values, so the second is a
    // replicate of the first: two points per cell.
    assert_eq!(a.len(), 12, "3 × 2 cells × 2 starts, got {}", a.len());
    let ids_a: BTreeSet<&String> = a.keys().collect();
    let ids_b: BTreeSet<&String> = b.keys().collect();
    assert_eq!(ids_a, ids_b, "reordering a listed grid must not change any run id");
    for (id, mle_a) in &a {
        assert_eq!(mle_a, &b[id],
            "point {id} stores a different mle.toml when the grid's values are reordered");
    }
}
