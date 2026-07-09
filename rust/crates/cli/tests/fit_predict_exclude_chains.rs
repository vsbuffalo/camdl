//! End-to-end acceptance for `camdl fit predict --exclude-chains` — read-side
//! chain selection over a posterior cloud.
//!
//! Proposal: docs/dev/proposals/2026-07-09-chain-selection-read-side.md
//!
//! Strategy: run one tiny real PGAS fit to produce a valid, self-contained fit
//! segment (archived IR, sidecar, FitView), then OVERWRITE its `draws.tsv` with
//! a controlled 4-chain cloud — three tight chains plus one deliberate outlier
//! (0-based chain 3 = the user's `chain 4`, with a much larger `beta`). This
//! gives a KNOWN outlier chain deterministically, which a converging sampler
//! would not reliably produce. `fit predict` then reads that cloud through the
//! one filter.

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
/// Columns are the exact model params (estimated first, then fixed), keyed by
/// the 0-based `chain` / `draw`. Values carry a small deterministic jitter so
/// the tight chains are not numerically identical (as real MCMC never is).
fn build_cloud() -> String {
    let mut s = String::from("chain\tdraw\tbeta\tgamma\tN0\tI0\trho\tk\n");
    for chain in 0..4 {
        for draw in 0..DRAWS_PER_CHAIN {
            let jitter = ((draw % 7) as f64 - 3.0) * 0.003; // in [-0.009, 0.009]
            let (beta, gamma) = if chain == 3 {
                (0.85 + jitter, 0.05) // the stuck outlier: fast epidemic
            } else {
                let base = 0.35 + (chain as f64 - 1.0) * 0.004; // 0.346 / 0.350 / 0.354
                (base + jitter, 0.14)
            };
            s.push_str(&format!(
                "{chain}\t{draw}\t{beta:.4}\t{gamma:.3}\t10000\t10\t0.6\t10\n"
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
