//! gh#506: an IF2 stage must start where `[estimate].start` says.
//!
//! The defect was at a CALLSITE, not inside a unit-testable function:
//! `fit/mod.rs` asked `FitRunConfig::build` for `random_starts` whenever the
//! stage had no upstream `starts_from` — i.e. on every scout stage — and that
//! overwrote every `EstimatedParam::initial` with a uniform draw in
//! `[lower, upper]`. `FitRunConfig::build(…, random_starts = false)` was
//! already correct in isolation, so no unit test on it could have caught this.
//! Hence an end-to-end pin: run a real fit and read the starting values back
//! out of the artifact the fit itself writes.
//!
//! `chain_starts.tsv` is the right artifact to assert on. It is the sidecar
//! `fit run` writes for exactly this audit question ("where did each chain
//! actually begin, before any perturbation"), and it is fed the same
//! `estimated_params` the chains are, so it cannot agree with the test while
//! disagreeing with the run.
//!
//! Kept cheap on purpose — 1 chain, 20 particles, 1 iteration. The starting
//! point is fixed before the first filter pass, so nothing here depends on the
//! fit converging, or even on it being a good fit.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn golden_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/seir_observations.ir.json")
}

/// Every `chain_starts.tsv` under `root`.
fn find_chain_starts(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "chain_starts.tsv") {
                found.push(p);
            }
        }
    }
    found
}

/// Parse `chain_starts.tsv` into (column names, one row of values per chain).
/// Comment lines (`#`) carry provenance, not data.
fn parse_chain_starts(path: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("header row")
        .split('\t').map(|s| s.to_string()).collect();
    let rows = lines.map(|l| {
        l.split('\t').skip(1) // drop the `chain` column
            .map(|c| c.parse::<f64>()
                .unwrap_or_else(|_| panic!("non-numeric start {c:?} in {l:?}")))
            .collect()
    }).collect();
    (header, rows)
}

#[test]
fn init_single_starts_every_chain_at_the_declared_start() {
    let bin = binary();
    assert!(bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display());
    let camdlc = camdlc_bin();
    assert!(camdlc.exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`", camdlc.display());

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let data = dir.join("obs.tsv");
    std::fs::write(&data,
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

    // beta's declared start sits well inside [0.01, 0.5] and far from both
    // bounds, so a uniform draw landing on it by chance is a ~2% event per
    // run at the 1e-9 tolerance below — and the seed is fixed, so it either
    // always passes or always fails, never flakes.
    let fit_toml = dir.join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

[estimate.beta]
bounds = [0.01, 0.5]
start  = 0.123

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 1
cooling    = 0.5
init       = "single"

[config]
dt = 1.0
"#, ir = golden_ir().display(), data = data.display())).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--seed", "1", "--allow-nonconverged-scout"])
        .output()
        .expect("spawn fit run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fit run failed:\nstderr={stderr}");

    let starts = find_chain_starts(&dir.join("results"));
    assert_eq!(starts.len(), 1,
        "expected exactly one chain_starts.tsv under the run tree, found {starts:?}");
    let (header, rows) = parse_chain_starts(&starts[0]);

    let beta_col = header.iter().position(|h| h == "beta")
        .unwrap_or_else(|| panic!("no beta column in {header:?}"));
    assert!(!rows.is_empty(), "chain_starts.tsv has no chain rows");
    for (i, row) in rows.iter().enumerate() {
        let got = row[beta_col - 1]; // header includes `chain`, rows dropped it
        assert!((got - 0.123).abs() < 1e-9,
            "chain {} started at beta={got}, not the declared [estimate].start \
             of 0.123. A value inside [0.01, 0.5] but not equal to 0.123 means \
             the start was overwritten by a bounds-uniform draw — gh#506.",
            i + 1);
    }
}
