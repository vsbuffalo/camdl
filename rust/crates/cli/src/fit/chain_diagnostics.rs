//! Per-chain log-likelihood outlier diagnostics for the Bayesian samplers.
//!
//! `fit summary`'s aggregate R̂/ESS answer "did the chains agree?" but not
//! "WHICH chain disagreed." When a minority of MCMC chains wander into a
//! non-representative side mode (a near-unidentified parameter carving out a
//! flat ridge), the user is left to open each `chain_N/trace.tsv` by hand to
//! find the stragglers. A per-chain robust z-score names them.
//!
//! Read-side only. Everything here is computed at `fit summary` time from the
//! committed trace files; nothing flows into a `run_id` / CAS key. Adding a
//! diagnostic does not re-key any fit.
//!
//! One shared seam. The gate message (`gating::format_decibans_spread_verdict`)
//! and the summary table (`fit_summary`) both name outliers via the same
//! [`chain_loglik_mod_zscores`] / [`outlier_labels`], so "what counts as an
//! outlier" is defined in exactly one place.
//!
//! ## The statistic: robust modified z-score (Iglewicz & Hoaglin 1993)
//!
//! We flag on the **modified z-score**
//!
//! ```text
//!   M_i = 0.6745 · (x_i − median(x)) / MAD ,    MAD = median(|x_i − median(x)|)
//! ```
//!
//! centred on the **median** and scaled by the **median absolute deviation**,
//! both of which ignore the outliers themselves. This is deliberately *not* the
//! classic `(x − mean)/SD` z-score: mean/SD let a minority of co-stuck chains
//! inflate the spread and mask their own flag (and cap a single outlier's |z|
//! at √(N−1), unreachable for small N). The robust version has neither problem
//! — a handful of chains in a side mode are flagged no matter how many share
//! it, because the median/MAD describe the *bulk*, not the whole set. The
//! `0.6745` (≈ Φ⁻¹(0.75)) rescales MAD to a consistent σ-estimator, so the
//! threshold stays comparable to a normal-theory z.
//!
//! `MAD == 0` (≥ half the chains share a numeric value, so the median has zero
//! robust spread — a degenerate case that does not arise for real MCMC means
//! but can in synthetic inputs) falls back to Iglewicz–Hoaglin's documented
//! alternate, the **mean absolute deviation**:
//!
//! ```text
//!   M_i = (x_i − mean(x)) / (1.253314 · MeanAD) ,   MeanAD = mean(|x_i − mean(x)|)
//! ```
//!
//! and if `MeanAD` is also 0 (every chain identical) there are no outliers.
//! Non-finite chain means (an unreadable trace) never poison the median/MAD —
//! they are excluded from the centre/scale and rendered as `—`, so a broken
//! chain is visible rather than silently reading as "clean."

use std::path::Path;

/// Flag a chain when its **modified z-score** exceeds this in magnitude. 3.5 is
/// the Iglewicz–Hoaglin (1993) recommended cutoff: with the 0.6745 rescaling,
/// MAD is a consistent σ-estimator, so 3.5 modified-SDs is the robust analogue
/// of a 3.5-σ classic z. Named (not inlined) so the gate and the summary share
/// one threshold.
pub const CHAIN_LOGLIK_OUTLIER_MODZ: f64 = 3.5;

/// One chain's mean log-likelihood and its robust modified z-score against the
/// between-chain spread.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainZScore {
    /// 1-based chain number for display (`chain 1` … `chain N`), matching the
    /// user-facing convention used everywhere else in fit output.
    pub chain: usize,
    /// Mean log-likelihood over this chain's post-burn-in draws. `NaN` when the
    /// chain had no readable trace rows (rendered `—`, never counted "clean").
    pub mean_loglik: f64,
    /// Modified z-score (median/MAD, or mean/MeanAD when MAD == 0). `0.0` for a
    /// single chain or perfect agreement; `NaN` when `mean_loglik` is non-finite.
    pub mod_z: f64,
    /// `|mod_z| > CHAIN_LOGLIK_OUTLIER_MODZ`. Always `false` for a non-finite
    /// `mod_z` (a NaN is surfaced as `—`, not swallowed as "not an outlier").
    pub is_outlier: bool,
}

/// Per-chain robust modified z-scores from per-chain mean logliks, in chain
/// order (index 0 = chain 1). See the module docs for the statistic. A single
/// chain, or perfect agreement, yields all-zero scores and nothing flagged —
/// the negative-control path.
pub fn chain_loglik_mod_zscores(means: &[f64]) -> Vec<ChainZScore> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    // Robust centre + scale from the FINITE means only: a NaN from an unreadable
    // trace must neither poison the median/MAD nor read as "clean."
    let finite: Vec<f64> = means.iter().copied().filter(|m| m.is_finite()).collect();
    let (center, scale) = robust_center_scale(&finite);
    means
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            // Finite mean + a positive, finite scale → a genuine z. Otherwise
            // (non-finite mean, or every finite chain identical so scale == 0)
            // there is no z to compute: NaN mean → NaN z (rendered `—`);
            // identical finite chains → 0.0.
            let mod_z = if m.is_finite() && scale.is_finite() && scale > 0.0 {
                (m - center) / scale
            } else if m.is_finite() {
                0.0
            } else {
                f64::NAN
            };
            // Explicit finiteness guard: a NaN z is NOT an outlier verdict, it is
            // "could not assess" — surfaced via `mean_loglik`/`mod_z` = `—`, never
            // silently passed as "not an outlier."
            let is_outlier = mod_z.is_finite() && mod_z.abs() > CHAIN_LOGLIK_OUTLIER_MODZ;
            ChainZScore {
                chain: i + 1,
                mean_loglik: m,
                mod_z,
                is_outlier,
            }
        })
        .collect()
}

/// `(center, scale)` for the modified z-score over the finite chain means.
/// Primary: `(median, MAD/0.6745)`. Fallback when `MAD == 0`:
/// `(mean, 1.253314·MeanAD)`. `scale == 0.0` (all values identical, or fewer
/// than one finite value) means "no spread" → every z is 0, nothing flagged.
fn robust_center_scale(finite: &[f64]) -> (f64, f64) {
    let n = finite.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let med = median_of(finite);
    let mad = median_of(&finite.iter().map(|x| (x - med).abs()).collect::<Vec<_>>());
    if mad > 0.0 {
        // M_i = 0.6745·(x − med)/MAD  ⇔  center = med, scale = MAD / 0.6745.
        (med, mad / 0.6745)
    } else {
        // MAD == 0: ≥ half the means share a value. Iglewicz–Hoaglin's alternate
        // scale is the mean absolute deviation: M_i = (x − mean)/(1.253314·MeanAD).
        let mean = finite.iter().sum::<f64>() / n as f64;
        let mean_ad = finite.iter().map(|x| (x - mean).abs()).sum::<f64>() / n as f64;
        (mean, 1.253314 * mean_ad)
    }
}

/// Median of a set of finite values (clones + sorts; the chain count is tiny).
/// Even length averages the two central order statistics. `NaN` for an empty set.
fn median_of(vals: &[f64]) -> f64 {
    let n = vals.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The flagged chains as `"chain N"` labels, worst-|mod_z| first. Empty when no
/// chain exceeds the threshold.
pub fn outlier_labels(scores: &[ChainZScore]) -> Vec<String> {
    let mut flagged: Vec<&ChainZScore> = scores.iter().filter(|s| s.is_outlier).collect();
    flagged.sort_by(|a, b| {
        b.mod_z
            .abs()
            .partial_cmp(&a.mod_z.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    flagged.iter().map(|s| format!("chain {}", s.chain)).collect()
}

/// Read every `chain_*/trace.tsv` under `stage_dir` and return the per-chain
/// mean post-burn-in log-likelihood, ordered by ascending chain number.
///
/// Returns `None` when no `chain_*/trace.tsv` files exist (the sampler wrote no
/// per-chain trace — e.g. an optimizer-only stage) so the caller can say so
/// rather than silently show an empty table.
///
/// ## Column
///
/// Every Bayesian sampler streams its trace through
/// [`super::trace_writer::TraceWriter`], whose fixed layout is
/// `<index>\t<loglik>\t log_posterior \t …`. So the log-likelihood is always
/// column index 1, whatever the sampler names it (`log_likelihood` for
/// PMMH/mh/nuts, `log_complete_data_ll` for PGAS). We key on that structural
/// invariant (guarded by checking column 2 is `log_posterior`) rather than a
/// per-sampler column name.
///
/// ## Post-burn-in selection
///
/// The post-burn-in kept draws are a suffix of the trace, and `draws.tsv`
/// records exactly how many there are per chain, so we take the **last `K_c`
/// rows** of chain `c`'s trace (`K_c` = that chain's row count in `draws.tsv`):
///
/// - **nuts** — the trace is written post-warmup only, `draws.tsv` has the same
///   `n_samples` rows: last `K_c` == all rows (exact).
/// - **pmmh / mh** — the trace carries a warm-up prefix (`step < burn_in`, one
///   row per step) then the thinned sampling suffix; `draws.tsv` counts the
///   suffix, so the last `K_c` rows are exactly the post-burn-in draws (exact).
/// - **pgas** — the trace carries every sweep (warm-up + all sampling,
///   unthinned); `draws.tsv` counts the thinned post-burn-in draws, so the last
///   `K_c` sweeps are a post-burn-in tail window (`K_c ≤ n_sweeps − burn_in`,
///   so warm-up is never included — a subset of the post-burn-in region when
///   thinning is on).
///
/// When `draws.tsv` is absent or unreadable (a partial / hand-built stage) we
/// fall back to the whole trace: without the posterior manifest we cannot strip
/// warm-up, so the mean is noisier but still honest. This never contaminates a
/// well-formed fit.
pub fn read_chain_mean_logliks(stage_dir: &Path) -> Option<Vec<f64>> {
    let mut chain_dirs = discover_chain_dirs(stage_dir);
    if chain_dirs.is_empty() {
        return None;
    }
    chain_dirs.sort_by_key(|(n, _)| *n);

    // Per-chain kept-draw counts from draws.tsv, in ascending chain-id order.
    // draws.tsv's `chain` column is 0-based and contiguous, so the i-th count
    // pairs with the i-th chain dir once both are sorted ascending.
    let draw_counts = read_draw_counts(&stage_dir.join("draws.tsv"));

    let mut out = Vec::with_capacity(chain_dirs.len());
    for (i, (_, dir)) in chain_dirs.iter().enumerate() {
        let all = read_trace_logliks(&dir.join("trace.tsv"));
        let selected: &[f64] = match draw_counts.get(i).copied() {
            Some(k) if k > 0 && k <= all.len() => &all[all.len() - k..],
            // No manifest for this chain (or a count that doesn't fit) — use the
            // whole trace rather than drop the chain.
            _ => &all,
        };
        let mean = if selected.is_empty() {
            f64::NAN
        } else {
            selected.iter().sum::<f64>() / selected.len() as f64
        };
        out.push(mean);
    }
    Some(out)
}

/// Collect `(chain_number, dir)` for every `chain_<N>` subdirectory of
/// `stage_dir` with a parseable trailing integer. The on-disk base differs by
/// sampler (nuts numbers from 0, pmmh/pgas from 1); sorting by the parsed
/// number gives a stable order either way, and the display index is assigned
/// sequentially by the caller.
fn discover_chain_dirs(stage_dir: &Path) -> Vec<(usize, std::path::PathBuf)> {
    let mut dirs = Vec::new();
    let Ok(entries) = std::fs::read_dir(stage_dir) else {
        return dirs;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix("chain_") {
            if let Ok(n) = rest.parse::<usize>() {
                dirs.push((n, path));
            }
        }
    }
    dirs
}

/// Read the log-likelihood column (structural column index 1) from a
/// `trace.tsv`. Empty when the file is missing/unreadable or the header doesn't
/// look like a `TraceWriter` trace (column 2 must be `log_posterior`).
fn read_trace_logliks(trace_path: &Path) -> Vec<f64> {
    let Ok(contents) = std::fs::read_to_string(trace_path) else {
        return Vec::new();
    };
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let cols: Vec<&str> = header.split('\t').collect();
    // Guard the structural invariant: <index> <loglik> log_posterior …
    if cols.len() < 3 || cols[2] != "log_posterior" {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        // Column 1 is the loglik regardless of the sampler's name for it.
        if let Some(v) = fields.nth(1).and_then(|s| s.parse::<f64>().ok()) {
            out.push(v);
        }
    }
    out
}

/// Read `draws.tsv` and return the per-chain kept-draw count, ordered by
/// ascending 0-based `chain` id. Empty when the file is missing or has no
/// `chain` column.
fn read_draw_counts(draws_path: &Path) -> Vec<usize> {
    let Ok(contents) = std::fs::read_to_string(draws_path) else {
        return Vec::new();
    };
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let cols: Vec<&str> = header.split('\t').collect();
    let Some(chain_col) = cols.iter().position(|c| *c == "chain") else {
        return Vec::new();
    };
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if let Some(id) = fields.get(chain_col).and_then(|s| s.parse::<usize>().ok()) {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    counts.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_a_single_clear_outlier() {
        // Five well-mixed chains near -100 (jittered, as real chains are), one
        // stuck far below at -400.
        let means = vec![-100.0, -101.0, -99.0, -100.5, -100.2, -400.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert_eq!(scores.len(), 6);
        let stuck = &scores[5];
        assert_eq!(stuck.chain, 6);
        assert!(stuck.is_outlier, "stuck chain must flag: mod_z = {}", stuck.mod_z);
        assert!(stuck.mod_z < -CHAIN_LOGLIK_OUTLIER_MODZ, "mod_z strongly negative: {}", stuck.mod_z);
        for sc in &scores[..5] {
            assert!(!sc.is_outlier, "well-mixed chain {} wrongly flagged (mod_z={})", sc.chain, sc.mod_z);
        }
        assert_eq!(outlier_labels(&scores), vec!["chain 6"]);
    }

    /// THE point of the robust upgrade: two co-stuck chains among six now BOTH
    /// flag (the classic mean/SD z masked them — they inflated the SD and pulled
    /// their own |z| below threshold). Good chains carry realistic jitter so the
    /// MAD is non-zero (real MCMC means are never numerically identical).
    #[test]
    fn four_good_two_stuck_flags_both_stuck_chains() {
        // 4 good (jittered) at ≈ -1204, 2 stuck at ≈ -1461.
        let means = vec![-1204.0, -1205.0, -1203.0, -1206.0, -1461.0, -1462.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[4].is_outlier, "stuck chain 5 must flag: mod_z={}", scores[4].mod_z);
        assert!(scores[5].is_outlier, "stuck chain 6 must flag: mod_z={}", scores[5].mod_z);
        assert!(
            scores[..4].iter().all(|s| !s.is_outlier),
            "good chains must not flag: {:?}",
            scores[..4].iter().map(|s| s.mod_z).collect::<Vec<_>>()
        );
        let labels = outlier_labels(&scores);
        assert!(labels.contains(&"chain 5".to_string()), "must name chain 5: {labels:?}");
        assert!(labels.contains(&"chain 6".to_string()), "must name chain 6: {labels:?}");
        assert_eq!(labels.len(), 2, "exactly the two stuck chains: {labels:?}");
    }

    #[test]
    fn flags_nothing_when_well_mixed() {
        // Negative control: all chains agree — no outlier, and the scores are
        // small and finite (not a vacuous "everything is NaN" pass).
        let means = vec![-100.0, -100.3, -99.8, -100.1, -99.9, -100.2];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores.iter().all(|s| !s.is_outlier), "well-mixed set must flag nothing");
        assert!(scores.iter().all(|s| s.mod_z.is_finite()), "scores must be finite");
        assert!(scores.iter().all(|s| s.mod_z.abs() < CHAIN_LOGLIK_OUTLIER_MODZ));
        assert!(outlier_labels(&scores).is_empty());
    }

    #[test]
    fn single_chain_has_no_spread() {
        let scores = chain_loglik_mod_zscores(&[-42.0]);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].mod_z, 0.0);
        assert!(!scores[0].is_outlier);
    }

    /// MAD == 0 (majority of chains numerically identical) must not divide by
    /// zero: it falls back to the MeanAD scale, which flags the lone outlier
    /// with a FINITE score (never NaN).
    #[test]
    fn mad_zero_meanad_fallback_flags_outlier_without_nan() {
        // Nine identical good chains at -50, one outlier at -500 → MAD = 0.
        let mut means = vec![-50.0; 9];
        means.push(-500.0);
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores.iter().all(|s| s.mod_z.is_finite()), "MeanAD fallback must yield finite scores, not NaN");
        assert!(scores[9].is_outlier, "the outlier must flag via MeanAD: mod_z={}", scores[9].mod_z);
        assert!(scores[..9].iter().all(|s| !s.is_outlier), "identical good chains must not flag");
        assert_eq!(outlier_labels(&scores), vec!["chain 10"]);
    }

    #[test]
    fn all_identical_chains_yield_zero_not_nan() {
        // Both MAD and MeanAD are 0 → scale 0 → z = 0, no outliers, no NaN.
        let scores = chain_loglik_mod_zscores(&[-10.0, -10.0, -10.0, -10.0]);
        assert!(scores.iter().all(|s| s.mod_z == 0.0), "identical chains → mod_z = 0, not NaN");
        assert!(scores.iter().all(|s| !s.is_outlier));
    }

    /// A chain with an unreadable trace (NaN mean) must not read as a clean
    /// "not an outlier" with a real score — its mod_z is NaN (rendered `—`), and
    /// it does not poison the finite chains' median/MAD.
    #[test]
    fn non_finite_mean_is_not_silently_clean() {
        let means = vec![-50.0, -51.0, -49.0, f64::NAN, -400.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[3].mod_z.is_nan(), "broken chain's mod_z must be NaN, not a fake number");
        assert!(!scores[3].is_outlier, "a NaN score is not an outlier VERDICT (it is 'unassessable')");
        // The real outlier among the finite chains is still flagged.
        assert!(scores[4].is_outlier, "the -400 outlier must still flag: mod_z={}", scores[4].mod_z);
    }

    /// End-to-end reader: synthesize a six-chain stage dir (`chain_N/trace.tsv`,
    /// one stuck) + a `draws.tsv` manifest, with a warm-up prefix in each trace
    /// that MUST be stripped. Good chains carry jitter (distinct means) so the
    /// robust MAD is non-zero.
    #[test]
    fn reader_strips_warmup_and_names_the_stuck_chain() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_reader");
        std::fs::create_dir_all(&dir).unwrap();

        let warmup = "1\t-900.0\t-905.0\n2\t-880.0\t-885.0\n";
        // dir numbering starts at 1 (pmmh/pgas convention). Distinct good means.
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            let body = format!("step\tlog_likelihood\tlog_posterior\n{warmup}{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        write_trace(1, "3\t-50.0\t-52.0\n4\t-50.0\t-52.0\n5\t-50.0\t-52.0\n"); // mean -50.0
        write_trace(2, "3\t-50.5\t-52.0\n4\t-50.5\t-52.0\n5\t-50.5\t-52.0\n"); // mean -50.5
        write_trace(3, "3\t-49.5\t-51.0\n4\t-49.5\t-51.0\n5\t-49.5\t-51.0\n"); // mean -49.5
        write_trace(4, "3\t-50.2\t-52.0\n4\t-50.2\t-52.0\n5\t-50.2\t-52.0\n"); // mean -50.2
        write_trace(5, "3\t-49.8\t-51.0\n4\t-49.8\t-51.0\n5\t-49.8\t-51.0\n"); // mean -49.8
        write_trace(6, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // stuck ≈ -300

        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let means = read_chain_mean_logliks(&dir).expect("chain traces present");
        assert_eq!(means.len(), 6);
        // Warm-up (-900/-880) excluded → good chains ≈ -50, not dragged to -890.
        for (i, m) in means.iter().take(5).enumerate() {
            assert!((m - (-50.0)).abs() < 1.0, "chain {} mean {} ≈ -50 (warm-up stripped)", i + 1, m);
        }
        assert!((means[5] - (-300.0)).abs() < 1.0, "chain 6 (stuck) mean {} ≈ -300", means[5]);

        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[5].is_outlier, "stuck chain 6 must flag: mod_z={}", scores[5].mod_z);
        assert!(scores[..5].iter().all(|s| !s.is_outlier), "good chains must not flag");
        assert_eq!(outlier_labels(&scores), vec!["chain 6"]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reader_returns_none_without_chain_dirs() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_chain_mean_logliks(&dir).is_none(), "no chain_* dirs → None");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reader_reads_pgas_column_name() {
        // PGAS names its loglik column `log_complete_data_ll`; the reader keys on
        // column position (1), not the name, so it must still read it.
        let dir = crate::test_support::unique_temp_dir("chain_diag_pgas");
        let cd = dir.join("chain_1");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(
            cd.join("trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\n0\t-77.0\t-79.0\n1\t-77.0\t-79.0\n",
        )
        .unwrap();
        let means = read_chain_mean_logliks(&dir).expect("one chain");
        assert!((means[0] - (-77.0)).abs() < 1e-9, "must read the pos-1 column: {}", means[0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
