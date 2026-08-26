//! gh#513: the preflight transforms table must report the start each chain
//! actually ran from, not the one written in the config.
//!
//! The defect was at a CALLSITE — `print_preflight` read
//! `config.estimated_params` while the chains ran from a per-chain override —
//! so, exactly as in gh#506, no unit test on the extracted helper can catch it.
//! A mutation check makes that concrete: reverting `print_preflight` to
//! `&config.estimated_params[..]` fully reintroduces gh#513, and every unit
//! test over `preflight_specs` still passes. Hence an end-to-end pin.
//!
//! The assertion is a CONSISTENCY one rather than a hardcoded draw: the value
//! the table prints must equal the value `chain_starts.tsv` records for chain
//! 1. Pinning the literal LHS draw would break the day the sampler's stream
//!
//! changes for an unrelated reason, and would not actually state the invariant,
//! which is that the two artifacts describe the same run. The declared start is
//! asserted ABSENT separately, so a table that silently fell back to the config
//! cannot pass by coincidence.
//!
//! `init = "lhs"` is the mode that separates the two: it stratifies over
//! `bounds` and uses the base point for no chain, so chain 1's start differs
//! from `[estimate].start`. Under `single`, or `uniform`'s chain 1, the two
//! coincide and the test would prove nothing.
//!
//! Cheap on purpose — 2 chains, 20 particles, 1 iteration. Starting points are
//! fixed before the first filter pass, so nothing depends on the fit
//! converging.

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

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() { break; }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// chain 1's `beta` from the run's `chain_starts.tsv`.
fn chain_one_beta(root: &Path) -> f64 {
    let mut stack = vec![root.to_path_buf()];
    let mut found = None;
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); }
            else if p.file_name().is_some_and(|n| n == "chain_starts.tsv") {
                assert!(found.is_none(), "more than one chain_starts.tsv under {}", root.display());
                found = Some(p);
            }
        }
    }
    let path = found.unwrap_or_else(|| panic!("no chain_starts.tsv under {}", root.display()));
    let txt = std::fs::read_to_string(&path).unwrap();
    let mut lines = txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = header.iter().position(|h| *h == "beta")
        .unwrap_or_else(|| panic!("no beta column in {header:?}"));
    let row: Vec<&str> = lines.next().expect("chain 1 row").split('\t').collect();
    row[col].parse().unwrap_or_else(|_| panic!("non-numeric beta {:?}", row[col]))
}

/// The value inside `log(...)` on the preflight table's `beta` row.
fn preflight_beta(stderr: &str) -> f64 {
    let line = stderr.lines()
        .find(|l| l.trim_start().starts_with("beta "))
        .unwrap_or_else(|| panic!("no beta row in the transforms table:\n{stderr}"));
    let open = line.find("log(").unwrap_or_else(|| panic!("no log(...) in {line:?}")) + 4;
    let close = line[open..].find(')').unwrap_or_else(|| panic!("unterminated log( in {line:?}")) + open;
    line[open..close].parse()
        .unwrap_or_else(|_| panic!("non-numeric preflight start {:?}", &line[open..close]))
}

#[test]
fn preflight_reports_the_start_the_chain_ran_from_not_the_configured_one() {
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
chains     = 2
particles  = 20
iterations = 1
cooling    = 0.5
init       = "lhs"

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
    let stderr = strip_ansi(&String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "fit run failed:\nstderr={stderr}");

    let shown = preflight_beta(&stderr);
    let ran_from = chain_one_beta(&dir.join("results"));

    // The invariant: the table and the audit sidecar describe one run.
    assert!((shown - ran_from).abs() < 1e-4,
        "the transforms table printed beta={shown}, but chain_starts.tsv records \
         chain 1 starting at {ran_from}. The table must report the realised \
         start (gh#513).\nstderr:\n{stderr}");

    // And it is genuinely the realised one, not the config's. Without this the
    // test passes against a table that fell back to `[estimate].start` on a run
    // where LHS happened to land near it.
    assert!((shown - 0.123).abs() > 1e-6,
        "the transforms table printed the declared [estimate].start (0.123) under \
         init = \"lhs\", which uses the base point for no chain — this is exactly \
         gh#513.\nstderr:\n{stderr}");

    // The header has to say the chains diverge, or a reader takes chain 1's
    // row for all of them.
    assert!(stderr.contains("chains start at different points"),
        "the transforms header must flag that chains start apart under lhs\
         \nstderr:\n{stderr}");
}
