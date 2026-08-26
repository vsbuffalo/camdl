//! Integration tests for the gh#74 Option B per-cell convergence
//! diagnostic columns emitted by `camdl profile`.
//!
//! Diagnostic columns are appended to the profile output TSV after the
//! pre-existing schema. Consumers must read by column name, not column
//! index. Algorithm-specific columns are present in every row; rows
//! produced by an algorithm that has no value for a given column write
//! `NaN` (capital N — matches camdl's TSV convention).
//!
//! Each test drives the release binary end-to-end, identical pattern
//! to `profile_priors.rs`. Skipped when the release binary or the
//! `camdlc` compiler isn't present.
//!
//! Schema asserted in these tests:
//!
//! * PMMH per-cell columns:
//!   `acc_rate_avg`, `acc_rate_min`,
//!   `loglik_spread_starts`, `loglik_rhat_starts`,
//!   `starts_n_completed`.
//! * IF2 per-cell columns:
//!   `iterations_used`, `cooling_final`,
//!   `loglik_spread_starts`, `loglik_rhat_starts`,
//!   `starts_n_completed`.
//! * NLopt per-cell columns: `loglik_spread_starts`,
//!   `starts_n_completed`.
//!
//! `loglik_rhat_starts` is NaN when fewer than 3 starts ran — Gelman-
//! Rubin Rhat is undefined / unstable at K<3.

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
impl Drop for Tmp {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_profile_diag_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// Tiny SIR fixture: two estimated params (`beta`, `gamma`), `N0`
/// fixed. Same shape used by `profile_priors.rs` so the test set
/// stays uniform.
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
scenarios {
  baseline {
    set = {
      beta  = 0.3
      gamma = 0.1
      N0    = 1000
    }
  }
}
init { S = 999  I = 1 }
simulate { from = 0 'days  to = 6 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path,
        "time\tcases\n1\t2\n2\t4\n3\t8\n4\t6\n5\t4\n6\t2\n").unwrap();

    (ir_path, data_path)
}

/// Parse the umbrella `summary.tsv` written under `<out_root>/profiles/`.
/// Returns (headers, data_rows). Comment lines are skipped.
fn read_summary_tsv(out_root: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let profiles = out_root.join("profiles");
    let entries: Vec<_> = std::fs::read_dir(&profiles)
        .unwrap_or_else(|e| panic!("read_dir {}: {}", profiles.display(), e))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    assert_eq!(entries.len(), 1,
        "expected one umbrella dir under {}, found {:?}",
        profiles.display(), entries);
    let umbrella = &entries[0];
    let summary = umbrella.join("summary.tsv");
    let body = std::fs::read_to_string(&summary)
        .unwrap_or_else(|e| panic!("read {}: {}", summary.display(), e));
    let mut lines = body.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header_line = lines.next()
        .unwrap_or_else(|| panic!("no header row in {}", summary.display()));
    let headers: Vec<String> = header_line.split('\t').map(|s| s.to_string()).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split('\t').map(|s| s.to_string()).collect())
        .collect();
    (headers, rows)
}

fn col_index(headers: &[String], name: &str) -> usize {
    headers.iter().position(|h| h == name)
        .unwrap_or_else(|| panic!(
            "header missing column '{}'. Available headers: {:?}",
            name, headers))
}

/// Parse a TSV cell as f64, accepting `NaN`/`Inf`/`-Inf` and bare
/// numerics. Used by the diagnostic assertions where columns can hold
/// either finite values or `NaN`.
fn parse_cell(s: &str) -> f64 {
    match s.trim() {
        "NaN" => f64::NAN,
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        v => v.parse::<f64>().unwrap_or_else(|e| {
            panic!("failed to parse TSV cell {:?}: {}", s, e)
        }),
    }
}

/// Common harness: run `camdl profile --algorithm <alg>` on the SIR
/// fixture with `--starts <k>`. Caller picks the algorithm-specific
/// hyperparams via `extra_args`.
fn run_profile_pmmh(
    bin: &Path, out_root: &Path, ir: &Path, data: &Path,
    starts: usize, seed: u64,
) -> std::process::Output {
    let out_tsv = out_root.join("profile.tsv");
    let args: Vec<String> = vec![
        "profile".into(), ir.to_string_lossy().into_owned(),
        "--scenario".into(), "baseline".into(),
        "--data".into(), data.to_string_lossy().into_owned(),
        "--obs".into(), "cases".into(),
        "--sweep".into(), "beta=lin(0.2,0.4,2)".into(),
        "--algorithm".into(), "pmmh".into(),
        // PMMH config has burn_in = 100 (see cli/src/profile.rs);
        // the loglik trace populates only post-burn-in. Use 200 so
        // we get ~100 post-burn samples → enough for Rhat at K>=3.
        "--pmmh-steps".into(), "200".into(),
        "--pmmh-particles".into(), "30".into(),
        "--pmmh-rho".into(), "0.99".into(),
        "--particles".into(), "30".into(),
        "--iterations".into(), "5".into(),
        "--starts".into(), format!("{}", starts),
        "--rw-sd".into(), "auto".into(),
        "--fixed".into(), "N0=1000".into(),
        "--output".into(), out_tsv.to_string_lossy().into_owned(),
        "--seed".into(), format!("{}", seed),
        "--suppress-warnings".into(),
    ];
    Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args.iter().map(|s| s.as_str()))
        .output()
        .expect("spawn camdl profile pmmh")
}

fn run_profile_if2(
    bin: &Path, out_root: &Path, ir: &Path, data: &Path,
    starts: usize, seed: u64,
) -> std::process::Output {
    let out_tsv = out_root.join("profile.tsv");
    let args: Vec<String> = vec![
        "profile".into(), ir.to_string_lossy().into_owned(),
        "--scenario".into(), "baseline".into(),
        "--data".into(), data.to_string_lossy().into_owned(),
        "--obs".into(), "cases".into(),
        "--sweep".into(), "beta=lin(0.2,0.4,2)".into(),
        "--algorithm".into(), "if2".into(),
        "--particles".into(), "30".into(),
        "--iterations".into(), "5".into(),
        "--starts".into(), format!("{}", starts),
        "--rw-sd".into(), "auto".into(),
        "--fixed".into(), "N0=1000".into(),
        "--output".into(), out_tsv.to_string_lossy().into_owned(),
        "--seed".into(), format!("{}", seed),
    ];
    Command::new(bin)
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args.iter().map(|s| s.as_str()))
        .output()
        .expect("spawn camdl profile if2")
}

// ─── Tests ──────────────────────────────────────────────────────────

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_pmmh_emits_acc_rate_columns() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("pmmh_acc_rate");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile_pmmh(&bin, &out_root, &ir, &data, 3, 1);
    assert!(output.status.success(),
        "profile pmmh failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));

    let (headers, rows) = read_summary_tsv(&out_root);
    assert!(!rows.is_empty(), "summary.tsv has no data rows");

    // Every required PMMH diagnostic column must be present.
    for col in ["acc_rate_avg", "acc_rate_min",
                "loglik_spread_starts", "loglik_rhat_starts",
                "starts_n_completed"] {
        assert!(headers.contains(&col.to_string()),
            "summary.tsv must declare column {:?}. headers={:?}",
            col, headers);
    }

    // Acceptance rates are finite probabilities for every cell.
    let i_avg = col_index(&headers, "acc_rate_avg");
    let i_min = col_index(&headers, "acc_rate_min");
    let i_completed = col_index(&headers, "starts_n_completed");
    for (ri, row) in rows.iter().enumerate() {
        let avg = parse_cell(&row[i_avg]);
        let min = parse_cell(&row[i_min]);
        let n_completed: f64 = parse_cell(&row[i_completed]);
        assert!(avg.is_finite() && (0.0..=1.0).contains(&avg),
            "row {} acc_rate_avg = {} must be a finite probability", ri, avg);
        assert!(min.is_finite() && (0.0..=1.0).contains(&min),
            "row {} acc_rate_min = {} must be a finite probability", ri, min);
        assert!(min <= avg + 1e-9,
            "row {} acc_rate_min ({}) must be <= acc_rate_avg ({})",
            ri, min, avg);
        // 3 starts requested; with --suppress-warnings & flat priors,
        // a 50-step PMMH chain should not diverge on this fixture.
        // The test asserts the column populates with an integer count,
        // not a particular value (induced-divergence is covered by a
        // separate test).
        assert!(n_completed >= 0.0,
            "row {} starts_n_completed = {} must be non-negative",
            ri, n_completed);
    }
}

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_pmmh_loglik_rhat_nan_at_k_lt_3() {
    // Gelman-Rubin R-hat is undefined / unstable for K < 3 chains.
    // With --starts 2 the column must hold NaN; the K<3 rule is
    // part of the documented schema.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("rhat_nan_k2");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile_pmmh(&bin, &out_root, &ir, &data, 2, 1);
    assert!(output.status.success(),
        "profile pmmh failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));

    let (headers, rows) = read_summary_tsv(&out_root);
    let i_rhat = col_index(&headers, "loglik_rhat_starts");
    for (ri, row) in rows.iter().enumerate() {
        let v = parse_cell(&row[i_rhat]);
        assert!(v.is_nan(),
            "row {} loglik_rhat_starts = {} must be NaN at K=2 (K<3 rule)",
            ri, v);
    }
}

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_pmmh_loglik_rhat_finite_at_k_3() {
    // At K=3 starts the K<3 rule lifts and R-hat must hold a finite
    // numeric value.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("rhat_finite_k3");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile_pmmh(&bin, &out_root, &ir, &data, 3, 1);
    assert!(output.status.success(),
        "profile pmmh failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));

    let (headers, rows) = read_summary_tsv(&out_root);
    let i_rhat = col_index(&headers, "loglik_rhat_starts");
    for (ri, row) in rows.iter().enumerate() {
        let v = parse_cell(&row[i_rhat]);
        assert!(v.is_finite(),
            "row {} loglik_rhat_starts = {} must be finite at K=3",
            ri, v);
    }
}

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_if2_emits_diagnostic_columns() {
    // IF2 path exposes per-cell iterations_used + cooling_final plus
    // the shared loglik_spread / loglik_rhat / starts_n_completed
    // columns. acc_rate_* is NaN for IF2 (the algorithm has no MH
    // acceptance step).
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("if2_diag");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let output = run_profile_if2(&bin, &out_root, &ir, &data, 3, 1);
    assert!(output.status.success(),
        "profile if2 failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));

    let (headers, rows) = read_summary_tsv(&out_root);
    for col in ["iterations_used", "cooling_final",
                "loglik_spread_starts", "loglik_rhat_starts",
                "starts_n_completed"] {
        assert!(headers.contains(&col.to_string()),
            "if2 summary.tsv must declare column {:?}. headers={:?}",
            col, headers);
    }

    let i_iters = col_index(&headers, "iterations_used");
    let i_cool  = col_index(&headers, "cooling_final");
    let i_completed = col_index(&headers, "starts_n_completed");
    for (ri, row) in rows.iter().enumerate() {
        let iters = parse_cell(&row[i_iters]);
        let cool  = parse_cell(&row[i_cool]);
        let n_completed = parse_cell(&row[i_completed]);
        assert!(iters.is_finite() && iters > 0.0,
            "row {} iterations_used = {} must be a positive finite count",
            ri, iters);
        assert!(cool.is_finite() && cool >= 0.0,
            "row {} cooling_final = {} must be a finite non-negative SD",
            ri, cool);
        assert!(n_completed.is_finite() && n_completed >= 0.0,
            "row {} starts_n_completed = {} must be a non-negative count",
            ri, n_completed);
    }
}

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_tsv_schema_stable_across_runs() {
    // Two runs with different seeds must produce a byte-identical
    // header. The schema doesn't reorder based on data.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("schema_stable");
    let (ir, data) = write_fixture(tmp.path());

    let out_a = tmp.path().join("out_a");
    let out_b = tmp.path().join("out_b");

    let out1 = run_profile_pmmh(&bin, &out_a, &ir, &data, 3, 1);
    assert!(out1.status.success(),
        "profile (seed=1) failed: stderr=\n{}",
        String::from_utf8_lossy(&out1.stderr));
    let out2 = run_profile_pmmh(&bin, &out_b, &ir, &data, 3, 42);
    assert!(out2.status.success(),
        "profile (seed=42) failed: stderr=\n{}",
        String::from_utf8_lossy(&out2.stderr));

    let (h_a, _) = read_summary_tsv(&out_a);
    let (h_b, _) = read_summary_tsv(&out_b);
    assert_eq!(h_a, h_b,
        "summary.tsv header must be byte-identical across runs differing \
         only by --seed. a={:?} b={:?}", h_a, h_b);
}

#[test]
#[ignore = "gh#154: asserts the per-cell diagnostic rollup (summary.tsv), the \
            deferred M4 derived view; raw per-start [diagnostics] survive per \
            leaf (see profile_leaf_mle_carries_per_start_diagnostics)"]
fn profile_starts_n_completed_reflects_diverged_chains() {
    // Plumbing test: assert `starts_n_completed` is wired through
    // the rollup. On the live fixture all three starts typically
    // complete (the PF returns very negative but still finite
    // logliks even under stress), so the end-to-end binary path can
    // only assert that the column populates with the K-or-less
    // count and never goes above K. The aggregator-level test
    // `aggregate_handles_diverged_chains` covers the < K case in
    // the unit suite where we can construct a `completed = false`
    // start record by hand.
    //
    // Why we don't do better via the binary path: divergence as
    // gh#74 defines it ("a start that completed without divergence")
    // requires the per-cell inference to *return* a non-finite final
    // loglik. For PMMH that requires every PF call across the chain
    // to fail (the chain falls back to the initial-params loglik
    // otherwise — see pmmh.rs:350-356). For IF2 it requires the IF2
    // engine itself to error or the clean-PF re-eval to return
    // -Inf. Both are rare on a small SIR fixture even with 2-particle
    // filters and extreme sweep values, so the integration test
    // exercises the "common path" (K everywhere) and the unit test
    // covers the < K case.
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("diverged");
    let (ir, data) = write_fixture(tmp.path());
    let out_root = tmp.path().join("out");

    let out_tsv = out_root.join("profile.tsv");
    let args: Vec<String> = vec![
        "profile".into(), ir.to_string_lossy().into_owned(),
        "--scenario".into(), "baseline".into(),
        "--data".into(), data.to_string_lossy().into_owned(),
        "--obs".into(), "cases".into(),
        "--sweep".into(), "beta=lin(2.5,4.5,2)".into(),
        "--algorithm".into(), "pmmh".into(),
        "--pmmh-steps".into(), "150".into(),
        "--pmmh-particles".into(), "2".into(),
        "--pmmh-rho".into(), "0.0".into(),
        "--particles".into(), "30".into(),
        "--iterations".into(), "5".into(),
        "--starts".into(), "3".into(),
        "--rw-sd".into(), "auto".into(),
        "--fixed".into(), "N0=1000".into(),
        "--output".into(), out_tsv.to_string_lossy().into_owned(),
        "--seed".into(), "7".into(),
        "--suppress-warnings".into(),
    ];
    let output = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args.iter().map(|s| s.as_str()))
        .output()
        .expect("spawn camdl profile");
    assert!(output.status.success(),
        "profile run failed: stderr=\n{}",
        String::from_utf8_lossy(&output.stderr));

    let (headers, rows) = read_summary_tsv(&out_root);
    let i_completed = col_index(&headers, "starts_n_completed");
    for row in &rows {
        let v = parse_cell(&row[i_completed]);
        assert!(v.is_finite() && (0.0..=3.0).contains(&v),
            "starts_n_completed must be a finite count in [0, K=3], got {}", v);
    }
}

/// Collect every `ProfilePoint` leaf's `mle.toml` body under `<out_root>`.
fn collect_leaf_mle(out_root: &Path) -> Vec<String> {
    fn walk(dir: &Path, out: &mut Vec<String>) {
        if dir.join("mle.toml").is_file() {
            if let Ok(b) = std::fs::read_to_string(dir.join("run.json")) {
                if b.contains("\"profile_point\"") {
                    if let Ok(m) = std::fs::read_to_string(dir.join("mle.toml")) {
                        out.push(m);
                    }
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() { if e.path().is_dir() { walk(&e.path(), out); } }
        }
    }
    let mut v = Vec::new();
    walk(&out_root.join("profiles"), &mut v);
    v
}

/// The cross-start aggregate rollup is deferred (gh#154), but the raw
/// per-start diagnostic data it aggregates is still written: each
/// `ProfilePoint` leaf's `mle.toml` carries a `[diagnostics]` block tagged
/// with the algorithm. This pins write→disk of that block — the input the
/// deferred M4 reindex re-aggregates — so the raw data can't silently vanish.
#[test]
fn profile_leaf_mle_carries_per_start_diagnostics() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("leaf_diag");
    let (ir, data) = write_fixture(tmp.path());

    // PMMH: the per-start [diagnostics] carries algorithm + acc_rate.
    let out_pmmh = tmp.path().join("out_pmmh");
    let o = run_profile_pmmh(&bin, &out_pmmh, &ir, &data, 1, 1);
    assert!(o.status.success(), "pmmh profile failed:\n{}",
        String::from_utf8_lossy(&o.stderr));
    let pmmh_mles = collect_leaf_mle(&out_pmmh);
    assert_eq!(pmmh_mles.len(), 2, "expected 2 pmmh leaves, got {}", pmmh_mles.len());
    for mle in &pmmh_mles {
        assert!(mle.contains("[diagnostics]"),
            "leaf mle.toml missing [diagnostics] block:\n{}", mle);
        assert!(mle.contains("algorithm = \"pmmh\""),
            "pmmh leaf [diagnostics] must tag algorithm = pmmh:\n{}", mle);
        assert!(mle.contains("acc_rate ="),
            "pmmh leaf [diagnostics] must record acc_rate:\n{}", mle);
    }

    // IF2: the per-start [diagnostics] is present and tagged if2.
    let out_if2 = tmp.path().join("out_if2");
    let o = run_profile_if2(&bin, &out_if2, &ir, &data, 1, 1);
    assert!(o.status.success(), "if2 profile failed:\n{}",
        String::from_utf8_lossy(&o.stderr));
    let if2_mles = collect_leaf_mle(&out_if2);
    assert_eq!(if2_mles.len(), 2, "expected 2 if2 leaves, got {}", if2_mles.len());
    for mle in &if2_mles {
        assert!(mle.contains("[diagnostics]"),
            "if2 leaf mle.toml missing [diagnostics] block:\n{}", mle);
        assert!(mle.contains("algorithm = \"if2\""),
            "if2 leaf [diagnostics] must tag algorithm = if2:\n{}", mle);
    }
}
