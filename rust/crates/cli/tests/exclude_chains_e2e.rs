//! End-to-end acceptance for read-side chain selection (`--exclude-chains`) on
//! `fit predict` and `fit summary`.
//!
//! Proposal: docs/dev/proposals/2026-07-09-chain-selection-read-side.md
//!
//! Strategy: run one tiny real PGAS fit to produce a valid, self-contained fit
//! segment (archived IR, sidecar, FitView), then OVERWRITE its `draws.tsv` with
//! a controlled 4-chain cloud — three tight chains plus one deliberate outlier
//! (0-based chain 3 = the user's `chain 4`, with a much larger `beta`). This
//! gives a KNOWN outlier chain deterministically, which a converging sampler
//! would not reliably produce. The read-side commands then read that cloud
//! through the one filter.

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
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  peak = max(I / N)   # value scalar — the outlier chain's fast epidemic peaks high
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str =
    "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// Draws per chain in the controlled cloud. Large enough that the q95 upper
/// band is a stable statistic (an extreme quantile over few draws is noisy), so
/// the well-mixed-vs-well-mixed control has a clean, low-noise baseline.
const DRAWS_PER_CHAIN: usize = 30;

/// A controlled 4-chain posterior cloud. Chains 0,1,2 (the user's 1,2,3) are
/// tight around a SLOW epidemic (`beta ≈ 0.35`); chain 3 (the user's `chain 4`)
/// is a stuck outlier at a FAST epidemic (`beta ≈ 0.85`), so dropping it must
/// move the free-forward bands sharply — and dropping a tight chain must not.
///
/// The tight chains carry a zero-mean cyclic jitter with a per-chain PHASE
/// shift: they are non-identical (as real MCMC is) but share the SAME mean, so
/// their between-chain variance is ~0 and the subset R̂ is healthy (~1). The
/// outlier is the only thing that inflates R̂ over the full cloud — exactly the
/// case `--exclude-chains` exists for. Columns are the exact model params
/// (estimated first, then fixed), keyed by the 0-based `chain` / `draw`.
fn build_cloud() -> String {
    let mut s = String::from("chain\tdraw\tbeta\tgamma\tN0\tI0\trho\tk\n");
    let n = DRAWS_PER_CHAIN;
    for chain in 0..4 {
        for draw in 0..n {
            // Cyclic phase shift: over a full period each chain sees the same
            // multiset of jitters → identical means, different sequences.
            let i = (draw + chain) % n;
            let jitter = ((i % 11) as f64 - 5.0) * 0.002; // in [-0.01, 0.01]
            let (beta, gamma) = if chain == 3 {
                (0.85 + jitter, 0.05) // the stuck outlier: fast epidemic
            } else {
                (0.35 + jitter, 0.14 + jitter * 0.1)
            };
            s.push_str(&format!(
                "{chain}\t{draw}\t{beta:.4}\t{gamma:.4}\t10000\t10\t0.6\t10\n"
            ));
        }
    }
    s
}

fn fit_toml() -> String {
    r#"output_dir = "results"

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

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 100
sweeps = 40
burn_in = 10
thin = 1
"#
    .to_string()
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The single fit segment directory under `results/fits/`.
fn segment_dir(root: &Path) -> PathBuf {
    let fits = root.join("fits");
    std::fs::read_dir(&fits)
        .expect("results/fits exists")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("one fit segment")
}

/// Recursively locate the single `draws.tsv` under a fit segment.
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

/// One free-forward predictive row: `(time, n_draws, q95)`. The `q95` upper
/// band is where a MINORITY outlier chain shows up strongly — a fast epidemic
/// pushes the top of the band far up, while it barely moves the median.
#[derive(Debug, Clone, Copy)]
struct FfRow {
    time: f64,
    n_draws: usize,
    q95: f64,
}

/// Parse the free-forward rows of `predictive/weekly_cases.tsv`.
fn free_forward_rows(pred_tsv: &str) -> Vec<FfRow> {
    let mut lines = pred_tsv.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|c| *c == name).unwrap();
    let (ti, hi, ni, q95i) = (col("time"), col("horizon"), col("n_draws"), col("q95"));
    let mut out = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f.get(hi).copied() != Some("free_forward") {
            continue;
        }
        out.push(FfRow {
            time: f[ti].parse().unwrap(),
            n_draws: f[ni].parse().unwrap(),
            q95: f[q95i].parse().unwrap(),
        });
    }
    out
}

/// Max relative q95 shift between two band sets over shared times, restricted to
/// times where the reference band is non-trivial (so integer jitter at tiny
/// counts does not dominate the ratio).
fn max_rel_q95_shift(a: &[FfRow], b: &[FfRow]) -> f64 {
    let mut m = 0.0_f64;
    for ra in a {
        if ra.q95 <= 20.0 {
            continue;
        }
        if let Some(rb) = b.iter().find(|r| (r.time - ra.time).abs() < 1e-9) {
            m = m.max((ra.q95 - rb.q95).abs() / ra.q95);
        }
    }
    m
}

/// Read the `rhat_max` / `ess_min` cells of the first free-forward row of a
/// `predictive/<stream>.tsv`. Both are section-level (identical across a
/// section's rows), so the first free-forward row is representative. Returns the
/// raw cell strings so an empty cell (NotAssessed) stays distinguishable from
/// `"0"`.
fn free_forward_convergence_cells(pred_tsv: &str) -> (String, String) {
    let mut lines = pred_tsv.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|c| *c == name).unwrap();
    let (hi, ri, ei) = (col("horizon"), col("rhat_max"), col("ess_min"));
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        if f.get(hi).copied() == Some("free_forward") {
            return (f[ri].to_string(), f[ei].to_string());
        }
    }
    panic!("no free_forward row in predictive tsv:\n{pred_tsv}");
}

/// Parse the `max R̂ = <x>` value from a `fit summary` (text) stdout.
fn summary_max_rhat(stdout: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.contains("max R̂ ="))
        .and_then(|l| l.split("max R̂ =").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("a max R̂ line in the posterior block")
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

/// Set up a fit segment whose `draws.tsv` is the controlled 4-chain cloud.
/// `label` keeps each test's tmp dir distinct (the tests run in parallel).
/// Returns `(tmp_dir, segment_dir)`.
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

    let seg = segment_dir(&tmp.join("results"));
    let draws = find_draws_tsv(&seg).expect("draws.tsv written by the fit");
    // Replace the sampler's cloud with the controlled one (same schema).
    std::fs::write(&draws, build_cloud()).unwrap();
    (tmp, seg)
}

#[test]
fn exclude_outlier_chain_moves_bands_and_is_recorded() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "outlier");
    let seg_str = seg.to_string_lossy().into_owned();

    // Full cloud (all 4 chains): free-forward predictive.
    let out = run(&bin, &tmp, &["fit", "predict", &seg_str, "--horizon", "free_forward"]);
    assert!(out.status.success(), "full predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let full = std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap();
    let full_rows = free_forward_rows(&full);
    assert!(!full_rows.is_empty(), "full predictive has free-forward rows");
    // n_draws over the full cloud = all 120 draws (4 chains × 30).
    assert!(full_rows.iter().all(|r| r.n_draws == 120), "full cloud replays 120 draws: {full_rows:?}");
    // predictive.json for the full run has NO chain_selection.
    let full_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(seg.join("predictive.json")).unwrap()).unwrap();
    assert!(
        full_json.get("chain_selection").is_none(),
        "a full-cloud predictive.json must not stamp chain_selection"
    );

    // Exclude the outlier chain 4 (0-based 3).
    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "4"],
    );
    assert!(out.status.success(), "excluded predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    // The non-quietable warning fired, naming the dropped chain.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--exclude-chains dropped chain(s) 4") && stderr.to_lowercase().contains("bias"),
        "predict must warn loudly about the biased selection, got:\n{stderr}"
    );

    let excl = std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap();
    let excl_rows = free_forward_rows(&excl);
    // n_draws dropped from 120 to 90 (chain 4's 30 draws gone).
    assert!(excl_rows.iter().all(|r| r.n_draws == 90), "excluded cloud replays 90 draws: {excl_rows:?}");

    // The bands MOVED, dramatically: the outlier chain's fast epidemic drives the
    // upper (q95) band far up at early times; dropping it collapses it. (The
    // median barely moves — the outlier is a minority — so q95 is the signal.)
    let outlier_effect = max_rel_q95_shift(&full_rows, &excl_rows);
    assert!(
        outlier_effect > 0.5,
        "dropping the outlier chain must move the free-forward q95 band a lot \
         (max rel shift {outlier_effect:.3})\nfull={full_rows:?}\nexcl={excl_rows:?}"
    );

    // Provenance stamped: chain_selection = {excluded:[4], kept:[1,2,3], n_total:4}.
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(seg.join("predictive.json")).unwrap()).unwrap();
    let cs = j.get("chain_selection").expect("chain-subset predictive.json stamps chain_selection");
    assert_eq!(cs["excluded"], serde_json::json!([4]));
    assert_eq!(cs["kept"], serde_json::json!([1, 2, 3]));
    assert_eq!(cs["n_total"], serde_json::json!(4));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn exclude_well_mixed_chain_barely_moves_but_is_recorded() {
    // Negative control: among the WELL-MIXED chains, dropping one more barely
    // moves the band — yet the selection is still recorded. The clean baseline
    // is "outlier removed" (chains 1,2,3); then also dropping the well-mixed
    // chain 2 (→ chains 1,3) must barely move the band, unlike dropping the
    // outlier. Comparing two clean subsets isolates the well-mixed effect
    // (dropping a tight chain while the outlier stays would instead RAISE the
    // outlier's weight — not a well-mixed control).
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "wellmixed");
    let seg_str = seg.to_string_lossy().into_owned();

    // Clean baseline: drop the outlier, keep the three tight chains.
    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "4"],
    );
    assert!(out.status.success(), "baseline exclude-4 failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let base_rows = free_forward_rows(
        &std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap(),
    );
    assert!(base_rows.iter().all(|r| r.n_draws == 90), "baseline keeps 3 tight chains (90 draws)");

    // Also drop a well-mixed chain (2). Chains 1,3 remain — still tight.
    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "2,4"],
    );
    assert!(out.status.success(), "well-mixed exclude failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let neg_rows = free_forward_rows(
        &std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap(),
    );
    assert!(neg_rows.iter().all(|r| r.n_draws == 60), "two tight chains remain (60 draws)");

    // The exclusion IS recorded — auditable regardless of effect size. Read the
    // manifest NOW, before the reference full-cloud run below overwrites it.
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(seg.join("predictive.json")).unwrap()).unwrap();
    let cs = j.get("chain_selection").expect("even a small-effect exclusion is stamped");
    assert_eq!(cs["excluded"], serde_json::json!([2, 4]));
    assert_eq!(cs["kept"], serde_json::json!([1, 3]));
    assert_eq!(cs["n_total"], serde_json::json!(4));

    // The full cloud (with the outlier) is the reference for the LARGE move.
    let out = run(&bin, &tmp, &["fit", "predict", &seg_str, "--horizon", "free_forward"]);
    assert!(out.status.success());
    let full_rows = free_forward_rows(
        &std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap(),
    );

    // Removing the OUTLIER (full → base) moves the q95 band a lot; removing a
    // WELL-MIXED chain (base → neg) moves it much less. The comparative claim is
    // robust to the few-draw y_rep noise that a fixed absolute threshold is not:
    // the outlier's fast epidemic inflates the upper band, a tight chain does not.
    let outlier_effect = max_rel_q95_shift(&full_rows, &base_rows);
    let wellmixed_effect = max_rel_q95_shift(&base_rows, &neg_rows);
    assert!(
        outlier_effect > 0.5,
        "removing the outlier chain must move the band clearly (got {outlier_effect:.3})"
    );
    assert!(
        wellmixed_effect < 0.3 && wellmixed_effect < outlier_effect,
        "removing a well-mixed chain must move the band far less than the outlier \
         (well-mixed {wellmixed_effect:.3} vs outlier {outlier_effect:.3})"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn nonexistent_chain_id_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "badid");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "9"],
    );
    assert!(!out.status.success(), "a chain id not in the fit must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("chain 9 not in this fit") && stderr.contains("chains 1..4"),
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
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "1,2,3,4"],
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

    // Text summary over the chain subset (drop the outlier chain 4).
    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "4", "--no-color"]);
    assert!(out.status.success(), "summary --exclude-chains failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Header names the subset and what was dropped.
    assert!(
        stdout.contains("chains:       3 of 4  (excluded 4)"),
        "summary header must show the chain subset, got:\n{stdout}"
    );
    // The recomputed R̂ over the three tight chains is finite and below the gate
    // (dropping the outlier is exactly the convergence-fixing move the feature
    // exists for; the stored full-cloud R̂ is replaced by the subset R̂).
    let rhat_line = stdout
        .lines()
        .find(|l| l.contains("max R̂ ="))
        .expect("a max R̂ line in the posterior block");
    let rhat: f64 = rhat_line
        .split("max R̂ =")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .expect("parse max R̂");
    assert!(rhat.is_finite() && rhat < 1.1, "subset R̂ recomputed and healthy: {rhat}");

    // The loud, non-quietable warning fired to stderr.
    assert!(
        stderr.contains("--exclude-chains dropped chain(s) 4") && stderr.to_lowercase().contains("bias"),
        "summary must warn about the biased selection, got:\n{stderr}"
    );

    // JSON summary stamps the provenance and carries the recomputed R̂.
    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--format", "json", "--exclude-chains", "4"]);
    assert!(out.status.success(), "json summary failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let j: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    let cs = j.get("chain_selection").expect("json summary stamps chain_selection");
    assert_eq!(cs["excluded"], serde_json::json!([4]));
    assert_eq!(cs["kept"], serde_json::json!([1, 2, 3]));
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
    // did). `fit summary` already recomputes over the subset; `fit predict` must
    // AGREE — that divergence IS the bug.
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "predict_subset_rhat");
    let seg_str = seg.to_string_lossy().into_owned();

    // Pin the stored full-cloud summary to a KNOWN, large R̂ / ESS, so the value
    // the buggy path copies is deterministic (the reported 1.570) and the
    // recomputed subset value is unambiguously different.
    let draws = find_draws_tsv(&seg).expect("draws.tsv written by the fit");
    let stage_dir = draws.parent().unwrap();
    std::fs::write(
        stage_dir.join("pgas_summary.json"),
        r#"{"stage":"pgas","n_chains":4,"thin":1,"rhat":{"beta":1.5700,"gamma":1.3000},"ess":{"beta":55,"gamma":47}}"#,
    )
    .unwrap();
    const STORED_RHAT: f64 = 1.5700;
    const STORED_ESS_MIN: f64 = 47.0;

    // Full cloud (no selection): the band carries the stored full-cloud summary.
    // That is CORRECT for the full cloud and must stay put — the regression guard.
    let out = run(&bin, &tmp, &["fit", "predict", &seg_str, "--horizon", "free_forward"]);
    assert!(out.status.success(), "full predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let (full_rhat_s, full_ess_s) = free_forward_convergence_cells(
        &std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap(),
    );
    let full_rhat: f64 = full_rhat_s.parse().unwrap();
    let full_ess: f64 = full_ess_s.parse().unwrap();
    assert!(
        (full_rhat - STORED_RHAT).abs() < 1e-9,
        "full-cloud band keeps the stored R̂ {STORED_RHAT}, got {full_rhat}"
    );
    assert!(
        (full_ess - STORED_ESS_MIN).abs() < 1e-9,
        "full-cloud band keeps the stored min ESS {STORED_ESS_MIN}, got {full_ess}"
    );

    // Exclude the stuck outlier chain 4: the band must be RELABELLED with the
    // recomputed subset R̂ / ESS.
    let out = run(
        &bin,
        &tmp,
        &["fit", "predict", &seg_str, "--horizon", "free_forward", "--exclude-chains", "4"],
    );
    assert!(out.status.success(), "excluded predict failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let (excl_rhat_s, excl_ess_s) = free_forward_convergence_cells(
        &std::fs::read_to_string(seg.join("predictive").join("weekly_cases.tsv")).unwrap(),
    );
    let excl_rhat: f64 = excl_rhat_s.parse().expect("recomputed rhat_max cell parses");
    let excl_ess: f64 = excl_ess_s.parse().expect("recomputed ess_min cell parses");

    // (1) Recomputed, not copied: strictly below the stored full-cloud R̂ (the
    //     outlier that inflated it is gone) and healthy (3 tight chains). On the
    //     buggy code excl_rhat == STORED_RHAT, so this assertion fails (RED).
    assert!(
        excl_rhat < STORED_RHAT - 1e-6,
        "subset R̂ must be recomputed below the stored full-cloud R̂ {STORED_RHAT} \
         — the bug copies {STORED_RHAT} into the subset band; got {excl_rhat}"
    );
    assert!(
        excl_rhat.is_finite() && excl_rhat < 1.1,
        "3 retained tight chains → healthy subset R̂: {excl_rhat}"
    );
    assert!(
        (excl_ess - STORED_ESS_MIN).abs() > 1e-6,
        "subset min ESS must be recomputed, not the stored {STORED_ESS_MIN}: got {excl_ess}"
    );

    // (2) predict and summary AGREE on the same fit + selection — the divergence
    //     that IS the bug is closed. summary recomputes over the same subset.
    let sout = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "4", "--no-color"]);
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
        stderr.contains("chain 9 not in this fit") && stderr.contains("chains 1..4"),
        "error names the bad id and range, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn summary_excluding_every_chain_hard_errors() {
    let bin = skip_if_missing_binary();
    let (tmp, seg) = setup(&bin, "summaryall");
    let seg_str = seg.to_string_lossy().into_owned();

    let out = run(&bin, &tmp, &["fit", "summary", &seg_str, "--exclude-chains", "1,2,3,4"]);
    assert!(!out.status.success(), "excluding every chain must be a hard error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("empty posterior"), "empty-posterior refusal, got:\n{stderr}");
    let _ = std::fs::remove_dir_all(&tmp);
}

// ── fit table --exclude-chains (forwarded to the --quantity derivation) ──────

/// Read the `peak` cell (JSON) for the single fit in a `fit table` run.
fn table_peak(bin: &Path, tmp: &Path, fits_root: &str, extra: &[&str]) -> (f64, String) {
    let mut args = vec!["fit", "table", fits_root, "--quantity", "peak", "--format", "json"];
    args.extend_from_slice(extra);
    let out = run(bin, tmp, &args);
    assert!(out.status.success(), "fit table failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let j: serde_json::Value = serde_json::from_str(&String::from_utf8_lossy(&out.stdout)).unwrap();
    // The document is an array of rows (or {rows:[...]}); find the one peak cell.
    let rows = j.get("rows").and_then(|r| r.as_array()).or_else(|| j.as_array()).expect("rows array");
    let peak = rows
        .iter()
        .find_map(|r| r.get("quantities").and_then(|q| q.get("peak")).and_then(|v| v.as_f64()))
        .expect("a peak cell");
    (peak, String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn table_forwards_selection_to_quantity_derivation() {
    let bin = skip_if_missing_binary();
    let (tmp, _seg) = setup(&bin, "table");
    let fits_root = tmp.join("results").join("fits");
    let fits_root_str = fits_root.to_string_lossy().into_owned();

    // Full cloud: peak median is tight (the outlier is a minority of draws).
    let (peak_full, _) = table_peak(&bin, &tmp, &fits_root_str, &[]);

    // Keep ONLY the outlier chain (drop the three tight chains): the peak median
    // jumps to the outlier's fast-epidemic peak — proof the flag reached the
    // derivation and reshaped the cloud.
    let (peak_outlier, stderr) = table_peak(&bin, &tmp, &fits_root_str, &["--exclude-chains", "1,2,3"]);
    assert!(
        peak_outlier > peak_full * 1.3,
        "keeping only the outlier chain must raise the derived peak median \
         (full {peak_full:.4} vs outlier-only {peak_outlier:.4})"
    );
    // The cohort warning fired.
    assert!(
        stderr.contains("will drop chain(s) 1,2,3") && stderr.to_lowercase().contains("bias"),
        "fit table must warn about the cohort-wide exclusion, got:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn table_exclude_chains_requires_quantity() {
    let bin = skip_if_missing_binary();
    let (tmp, _seg) = setup(&bin, "tablenoq");
    let fits_root = tmp.join("results").join("fits");
    let fits_root_str = fits_root.to_string_lossy().into_owned();

    // --exclude-chains without --quantity is inert (fit table otherwise reads
    // stored metadata), so clap refuses it rather than silently doing nothing.
    let out = run(&bin, &tmp, &["fit", "table", &fits_root_str, "--exclude-chains", "4"]);
    assert!(!out.status.success(), "--exclude-chains without --quantity must be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("quantity") || stderr.contains("required"),
        "clap error points at the missing --quantity, got:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
