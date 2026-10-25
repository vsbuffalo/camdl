//! Determinism PIN tests — the tripwire for the simulate/batch `run_job`
//! unification (proposal: 2026-05-28-simulate-batch-coherence-and-obs-ensembles.md).
//!
//! These lock the parts of `simulate`'s current RNG behaviour that the
//! existing `ir/golden` single-run determinism suites do NOT cover, and that
//! a careless engine reroute would silently break (CLAUDE.md §"RNG and
//! paired-seed coupling"):
//!
//!   1. Scenario CRN coupling — scenario index is deliberately absent from
//!      the seed mix (`main.rs:833-841`), so an `enable`/`disable` scenario
//!      shares the baseline's randomness byte-for-byte until the intervention
//!      fires. A reroute that folded scenario into the seed would destroy
//!      this (scenario comparisons become noise — silent-wrong-answer class).
//!   2. Seed coherence — "seed N" must mean the same trajectory whether run
//!      alone (`--seed N`) or inside a multi-run (`--seeds …,N,…`).
//!   3. Determinism — identical invocation ⇒ byte-identical output.
//!
//! All three MUST be green on the current (pre-reroute) code. They are the
//! reference the unified engine must reproduce. If a reroute turns one red,
//! the draw order / seed derivation diverged — stop and find the cause; do
//! NOT relax the assertion.

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

/// SIR + V with a vaccination intervention (S→V transfer) that fires at
/// t=15, inactive by default. Enabling it must leave the t<15 trajectory
/// byte-identical to baseline (CRN coupling) and diverge from t≥15.
fn write_model(path: &Path) {
    let src = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta  : rate in [0.01, 5.0]
  gamma : rate in [0.01, 5.0]
}

init { S = 990  I = 10  R = 0  V = 0 }

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

simulate { from = 0 'days  to = 30 'days }

interventions {
  vacc_campaign : transfer(fraction = 0.5, from = S, to = V) at [15]
}
"#;
    std::fs::write(path, src).unwrap();
}

/// Non-comment, non-blank lines of a TSV.
fn content_lines(tsv: &str) -> Vec<&str> {
    tsv.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

/// Map a single-run trajectory (columns `t  S  I  R  V  flow_*`) to t → full row.
fn rows_by_t(tsv: &str) -> BTreeMap<i64, String> {
    let lines = content_lines(tsv);
    let mut out = BTreeMap::new();
    for l in lines.iter().skip(1) {
        // skip header
        let t: i64 = l.split('\t').next().unwrap().parse().unwrap();
        out.insert(t, l.to_string());
    }
    out
}

fn run_sim(bin: &Path, model: &Path, extra: &[&str], out: &Path) {
    let mut args: Vec<String> = vec![
        "simulate".into(), model.to_string_lossy().into_owned(),
        "--param".into(), "beta=0.4".into(),
        "--param".into(), "gamma=0.2".into(),
        "--backend".into(), "chain_binomial".into(),
        "--dt".into(), "1".into(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    args.push("-o".into());
    args.push(out.to_string_lossy().into_owned());
    let o = Command::new(bin).args(&args).output().expect("spawn");
    assert!(o.status.success(),
        "simulate {:?} failed: {}", extra, String::from_utf8_lossy(&o.stderr));
}

/// PIN 1 — scenario CRN coupling. Baseline and `--enable vacc_campaign` at the
/// same seed must be byte-identical for every t before the intervention fires
/// (t<15) and must diverge afterward (proving the intervention is real, not a
/// no-op that makes the test pass vacuously).
#[test]
fn crn_coupling_pre_intervention_byte_identical() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    write_model(&model);

    let base = tmp.path().join("base.tsv");
    let intv = tmp.path().join("intv.tsv");
    run_sim(&bin, &model, &["--seed", "5"], &base);
    run_sim(&bin, &model, &["--seed", "5", "--enable", "vacc_campaign"], &intv);

    let b = rows_by_t(&std::fs::read_to_string(&base).unwrap());
    let i = rows_by_t(&std::fs::read_to_string(&intv).unwrap());
    assert_eq!(b.keys().collect::<Vec<_>>(), i.keys().collect::<Vec<_>>(),
        "baseline and intervened must cover the same timepoints");

    let mut diverged_after = false;
    for (t, brow) in &b {
        let irow = &i[t];
        if *t < 15 {
            assert_eq!(brow, irow,
                "CRN coupling broken at t={t}: enabling an intervention that \
                 fires at t=15 changed the pre-intervention trajectory. The \
                 same seed must consume the RNG identically until the \
                 intervention modifies state.\n  baseline:  {brow}\n  intervened:{irow}");
        } else if brow != irow {
            diverged_after = true;
        }
    }
    assert!(diverged_after,
        "intervened run never diverged from baseline at t>=15 — the \
         intervention did not fire, so this test is not actually exercising \
         CRN coupling. Fix the fixture.");
}

/// PIN 2 — seed coherence. `--seeds 1,2,3` is a multi-run; its replicate N
/// (1-based, positional with the seed list) must reproduce the standalone
/// `--seed <that value>` trajectory exactly. "Seed 2" must mean one thing.
#[test]
fn explicit_seeds_match_single_runs() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    write_model(&model);

    let multi = tmp.path().join("multi.tsv");
    run_sim(&bin, &model, &["--seeds", "7,8,9"], &multi);

    // Multi output has a leading `replicate` column. Extract replicate 1's
    // rows (0-based → the second seed, 8), stripping the replicate column,
    // into t → row.
    let multi_txt = std::fs::read_to_string(&multi).unwrap();
    let lines = content_lines(&multi_txt);
    let header = lines[0];
    assert!(header.starts_with("replicate\t"),
        "multi-run output should lead with a `replicate` column: {header}");
    let mut rep2: BTreeMap<i64, String> = BTreeMap::new();
    for l in lines.iter().skip(1) {
        let mut it = l.splitn(2, '\t');
        let rep: u64 = it.next().unwrap().parse().unwrap();
        let rest = it.next().unwrap();
        if rep == 1 {
            let t: i64 = rest.split('\t').next().unwrap().parse().unwrap();
            rep2.insert(t, rest.to_string());
        }
    }
    assert!(!rep2.is_empty(), "no replicate-1 rows found in multi output");

    // Standalone seed 8 (replicate 1 ↔ second seed in the list).
    let single = tmp.path().join("single8.tsv");
    run_sim(&bin, &model, &["--seed", "8"], &single);
    let single_rows = rows_by_t(&std::fs::read_to_string(&single).unwrap());

    assert_eq!(rep2, single_rows,
        "replicate 1 of `--seeds 7,8,9` must be byte-identical to standalone \
         `--seed 8`. A mismatch means the multi-run path derives or threads \
         the seed differently from the single-run path — exactly the \
         incoherence the unification must not introduce.");
}

/// PIN 3 — determinism. The same multi-run invocation twice must produce
/// byte-identical output (data region). Catches any nondeterminism a reroute
/// might introduce (e.g. iteration over a HashMap, time-seeded fallback).
#[test]
fn same_invocation_is_byte_identical() {
    let bin = skip_if_missing_binary();
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    write_model(&model);

    let a = tmp.path().join("a.tsv");
    let b = tmp.path().join("b.tsv");
    run_sim(&bin, &model, &["--seed", "42", "--replicates", "3"], &a);
    run_sim(&bin, &model, &["--seed", "42", "--replicates", "3"], &b);

    let la = content_lines(&std::fs::read_to_string(&a).unwrap()).join("\n");
    let lb = content_lines(&std::fs::read_to_string(&b).unwrap()).join("\n");
    assert_eq!(la, lb,
        "identical `simulate --seed 42 --replicates 3` invocations produced \
         different output — the run is not deterministic.");
}
