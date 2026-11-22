//! End-to-end acceptance for read-side chain selection (`--exclude-chains`) on
//! `camdl predict` and `fit summary`.
//!
//! Proposal: docs/dev/proposals/2026-07-09-chain-selection-read-side.md
//!
//! Each test runs one tiny real four-chain PGAS fit, stored in its work
//! dir's store, and reads its posterior through the read-side commands.
//! Draws a stored run holds cannot be replaced, so the tests here check what
//! a selection is recorded as, what it refuses, and that `camdl predict` and
//! `fit summary` recompute their convergence numbers over the same subset.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

/// The latest `camdl predict` of the run `run` stored: the prediction under
/// `run/predict/` whose `run.json` finished last. A path that does not exist
/// while nothing has been predicted.
fn predict_out(run: &Path) -> PathBuf {
    fit_store_support::predictions(run).into_iter()
        .max_by_key(|p| fit_store_support::json(&p.join("run.json"))["finished"]
            .as_str().unwrap_or_default().to_string())
        .unwrap_or_else(|| run.join("predict").join("none"))
}

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

/// A closed SIR with a weekly NegBinomial observation — small and well-behaved.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  peak = max(I / N)   # value scalar
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml() -> String {
    r#"store = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

[method]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 100
sweeps = 160
burn_in = 100
thin = 1
"#
    .to_string()
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_STORE", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// Recursively locate the single `draws.tsv` under a fit run folder.
fn find_draws_tsv(dir: &Path) -> Option<PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = find_draws_tsv(&p) {
                return Some(found);
            }
        } else if p.file_name().and_then(|n| n.to_str()) == Some("draws.tsv") {
            return Some(p);
        }
    }
    None
}

/// Read the `fit_rhat_max` / `fit_ess_min` cells of the first free-forward row of a
/// `predictive/<stream>.tsv`. Both are section-level (identical across a
/// section's rows), so the first free-forward row is representative. Returns the
/// raw cell strings so an empty cell (NotAssessed) stays distinguishable from
/// `"0"`.
fn free_forward_convergence_cells(pred_tsv: &str) -> (String, String) {
    let mut lines = pred_tsv.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|c| *c == name).unwrap();
    let (hi, ri, ei) = (col("horizon"), col("fit_rhat_max"), col("fit_ess_min"));
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f.get(hi).copied() == Some("free_forward") {
            return (f[ri].to_string(), f[ei].to_string());
        }
    }
    panic!("no free_forward row in predictive tsv:\n{pred_tsv}");
}

/// Parse the `max R̂ <x> (<param>), threshold <t>` value from a `fit summary`
/// (text) stdout — the verdict's second line.
fn summary_max_rhat(stdout: &str) -> f64 {
    stdout
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("max R\u{302} "))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("a max R̂ line in the verdict")
}

/// Parse the `min-param ESS <n>` token from a `fit summary` (text) stdout (the
/// ESS/iter line; ESS/sec repeats the same value).
fn summary_min_ess(stdout: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.contains("min-param ESS"))
        .and_then(|l| l.split("min-param ESS").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("a min-param ESS token in the ESS/iter line")
}

/// Run the four-chain PGAS fit in a fresh tmp dir and return the dir and
/// the stored run's folder. `label` keeps each test's tmp dir distinct (the
/// tests run in parallel).
fn setup(bin: &Path, label: &str) -> (PathBuf, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("camdl_excl_chains_{}_{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("model.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), fit_toml()).unwrap();

    let out = run(bin, &tmp, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert!(
        out.status.success(),
        "fit run failed:\nstderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = fit_store_support::the_run(&tmp.join("results"));
    assert!(find_draws_tsv(&run).is_some(), "draws.tsv written by the fit");
    (tmp, run)
}

#[test]
fn nonexistent_chain_id_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "badid");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(
        &bin,
        &tmp,
        &["predict", &seg_str, "--mode", "free_forward", "--exclude-chains", "9"],
    );
    assert!(!out.status.success(), "a chain id not in the fit must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("chain 9 not in this fit") && stderr.contains("chains 0..3"),
        "error names the bad id and the valid range, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn excluding_every_chain_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "excludeall");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(
        &bin,
        &tmp,
        &["predict", &seg_str, "--mode", "free_forward", "--exclude-chains", "0,1,2,3"],
    );
    assert!(!out.status.success(), "excluding every chain must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("empty posterior"),
        "error explains the empty-posterior refusal, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

// ── fit summary --exclude-chains ────────────────────────────────────────────

#[test]
fn summary_subset_shows_header_recomputes_and_warns() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "summary");
    let seg_str = seg.to_string_lossy().into_owned();

    // Text summary over the chain subset (drop chain 3).
    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "3", "--no-color"]);
    assert!(out.status.success(), "summary --exclude-chains failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // The method's identity line names the subset and what was dropped.
    assert!(
        stdout.contains("3 of 4 chains (excluded 3)"),
        "summary identity line must show the chain subset, got:\n{stdout}"
    );
    // And the per-chain table says what the selection did NOT filter, so its
    // rows are not read as sharing the recomputed scope of the tables above.
    assert!(
        stdout.contains("including the excluded 3"),
        "the chains table must name the scope it kept, got:\n{stdout}"
    );
    // The R̂ over the three retained chains is recomputed: it differs from
    // the stored full-cloud R̂.
    let rhat = summary_max_rhat(&stdout);
    let full = run(&bin, &tmp, &["fit", "summary", &seg_str, "--no-color"]);
    assert!(full.status.success());
    let full_rhat = summary_max_rhat(&String::from_utf8_lossy(&full.stdout));
    assert!(rhat.is_finite() && (rhat - full_rhat).abs() > 1e-6,
        "subset R̂ recomputed: subset {rhat}, full cloud {full_rhat}");

    // The loud, non-quietable warning fired to stderr.
    assert!(
        stderr.contains("--exclude-chains dropped chain(s) 3") && stderr.to_lowercase().contains("bias"),
        "summary must warn about the biased selection, got:\n{stderr}"
    );

    // JSON summary stamps the provenance and carries the recomputed R̂.
    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--format", "json", "--exclude-chains", "3"]);
    assert!(out.status.success(), "json summary failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let j: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    let cs = j.get("chain_selection").expect("json summary stamps chain_selection");
    assert_eq!(cs["excluded"], serde_json::json!([3]));
    assert_eq!(cs["kept"], serde_json::json!([0, 1, 2]));
    assert_eq!(cs["n_total"], serde_json::json!(4));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn summary_full_cloud_is_unchanged_regression() {
    // No selection: the summary is byte-identical to before this feature — no
    // header change, no chain_selection field, no warning.
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "summaryfull");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--no-color"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("(excluded"), "no exclusion header without the flag:\n{stdout}");

    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--format", "json"]);
    assert!(out.status.success());
    let j: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    assert!(j.get("chain_selection").is_none(), "no chain_selection field without the flag");

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn predict_subset_recomputes_convergence_agreeing_with_summary() {
    // gh#409: a chain-subset predictive band must carry the R̂ / ESS of the
    // RETAINED chains, not the stored full-cloud summary (which includes the
    // dropped chains). The buggy path copied the stored value into every row of
    // `predictive/*.tsv` — labelling a band drawn from the clean chains with the
    // polluted full-cloud R̂ (a "did not converge" verdict about a subset that
    // did). `fit summary` already recomputes over the subset; `camdl predict` must
    // AGREE — that divergence IS the bug.
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "predict_subset_rhat");
    let seg_str = seg.to_string_lossy().into_owned();

    // The stored full-cloud summary: the values the buggy path copies.
    let summary = fit_store_support::json(&seg.join("pgas_summary.json"));
    let max_of = |m: &serde_json::Value, f: fn(f64, f64) -> f64, init: f64| m.as_object()
        .expect("a per-parameter table").values().map(|v| v.as_f64().unwrap()).fold(init, f);
    let stored_rhat = max_of(&summary["rhat"], f64::max, f64::NEG_INFINITY);
    let stored_ess_min = max_of(&summary["ess"], f64::min, f64::INFINITY);

    // Full cloud (no selection): the band carries the stored full-cloud summary.
    // That is CORRECT for the full cloud and must stay put — the regression guard.
    let out = run(&bin, &tmp, &["predict", &seg_str, "--mode", "free_forward"]);
    assert!(out.status.success(), "full predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let (full_rhat_s, full_ess_s) = free_forward_convergence_cells(
        &std::fs::read_to_string(predict_out(&seg).join("predictive").join("weekly_cases.tsv")).unwrap(),
    );
    let full_rhat: f64 = full_rhat_s.parse().unwrap();
    let full_ess: f64 = full_ess_s.parse().unwrap();
    assert!(
        (full_rhat - stored_rhat).abs() < 1e-3,
        "full-cloud band keeps the stored R̂ {stored_rhat}, got {full_rhat}"
    );
    assert!(
        (full_ess - stored_ess_min).abs() < 1.5,
        "full-cloud band keeps the stored min ESS {stored_ess_min}, got {full_ess}"
    );

    // Exclude chain 3: the band must be RELABELLED with the
    // recomputed subset R̂ / ESS.
    let out = run(
        &bin,
        &tmp,
        &["predict", &seg_str, "--mode", "free_forward", "--exclude-chains", "3"],
    );
    assert!(out.status.success(), "excluded predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let (excl_rhat_s, excl_ess_s) = free_forward_convergence_cells(
        &std::fs::read_to_string(predict_out(&seg).join("predictive-excl3").join("weekly_cases.tsv")).unwrap(),
    );
    let excl_rhat: f64 = excl_rhat_s.parse().expect("recomputed rhat_max cell parses");
    let excl_ess: f64 = excl_ess_s.parse().expect("recomputed ess_min cell parses");

    // (1) Recomputed, not copied: the subset values differ from the stored
    //     full-cloud ones. On the buggy code they are equal (RED).
    assert!(
        excl_rhat.is_finite() && (excl_rhat - stored_rhat).abs() > 1e-6,
        "subset R̂ must be recomputed, not the stored full-cloud R̂ {stored_rhat}: got {excl_rhat}"
    );
    assert!(
        (excl_ess - stored_ess_min).abs() > 1e-6,
        "subset min ESS must be recomputed, not the stored {stored_ess_min}: got {excl_ess}"
    );

    // (2) predict and summary AGREE on the same fit + selection — the divergence
    //     that IS the bug is closed. summary recomputes over the same subset.
    let sout = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "3", "--no-color"]);
    assert!(sout.status.success(), "summary --exclude-chains failed:\n{}", String::from_utf8_lossy(&sout.stderr));
    let sstdout = String::from_utf8_lossy(&sout.stdout);
    let summary_rhat = summary_max_rhat(&sstdout);
    let summary_ess = summary_min_ess(&sstdout);
    assert!(
        (excl_rhat - summary_rhat).abs() < 1e-3,
        "predict and summary must report the SAME subset R̂: predict {excl_rhat} vs summary {summary_rhat}"
    );
    assert!(
        (excl_ess - summary_ess).abs() < 1.5,
        "predict and summary must report the SAME subset min ESS: predict {excl_ess} vs summary {summary_ess}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn summary_nonexistent_chain_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "summarybadid");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "9"]);
    assert!(!out.status.success(), "a chain id not in the fit must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("chain 9 not in this fit") && stderr.contains("chains 0..3"),
        "error names the bad id and range, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn summary_excluding_every_chain_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "summaryall");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "0,1,2,3"]);
    assert!(!out.status.success(), "excluding every chain must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("empty posterior"), "empty-posterior refusal, got:\n{stderr}");
    let _ = std::fs::remove_dir_all(&tmp);
}

// ── fit table --exclude-chains (forwarded to the --quantity derivation) ──────

#[test]
fn table_exclude_chains_requires_quantity() {
    let bin = skip_if_missing_binary();
    let (tmp, _seg) = setup(&bin, "tablenoq");
    let fits_root_str = tmp.join("results").to_string_lossy().into_owned();

    // --exclude-chains without --quantity is inert (fit table otherwise reads
    // stored metadata), so clap refuses it rather than silently doing nothing.
    let out = run(&bin, &tmp, &["fit", "table", &fits_root_str, "--exclude-chains", "3"]);
    assert!(!out.status.success(), "--exclude-chains without --quantity must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("quantity") || stderr.contains("required"),
        "clap error points at the missing --quantity, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
