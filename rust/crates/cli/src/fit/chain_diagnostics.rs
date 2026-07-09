//! Per-chain log-likelihood outlier diagnostics for the Bayesian samplers.
//!
//! `fit summary`'s aggregate R̂/ESS answer "did the chains agree?" but not
//! "WHICH chain disagreed." When a minority of MCMC chains wander into a
//! non-representative side mode (a near-unidentified parameter carving out a
//! flat ridge), the user is left to open each `chain_N/trace.tsv` by hand to
//! find the stragglers. A per-chain mean-loglik z-score names them.
//!
//! Read-side only. Everything here is computed at `fit summary` time from the
//! committed trace files; nothing flows into a `run_id` / CAS key. Adding a
//! diagnostic does not re-key any fit.
//!
//! One shared seam. The gate message (`gating::format_decibans_spread_verdict`)
//! and the summary table (`fit_summary`) both name outliers via the same
//! [`chain_loglik_zscores`] / [`outlier_labels`], so "what counts as an
//! outlier" is defined in exactly one place.
//!
//! Known limitations of the classic z-score (documented, not hidden). The
//! statistic is `(chain_mean − grand_mean) / SD(chain means)` with the
//! *population* SD of the N chain means (they are the whole population of
//! chains, not a sample from a superpopulation, so the population SD is the
//! honest descriptor — and it makes the threshold reachable at the chain counts
//! that occur in practice). Two consequences the reader should know:
//!
//!   1. **Small-N ceiling.** A single outlier's |z| is bounded by `√(N−1)`, so
//!      the 2.0 threshold is only reachable with `N ≥ 5` chains. With ≤4 chains
//!      the automatic flag cannot fire; the per-chain *table* is then the signal.
//!   2. **Masking.** Two or more chains sharing a side mode inflate the SD and
//!      can pull their own |z| below the threshold. The table still shows every
//!      chain's mean + z, so co-stuck chains are visible even when the automatic
//!      flag misses one.
//!
//! A robust modified z-score (median / MAD, Iglewicz & Hoaglin 1993) removes
//! both limitations and is the natural follow-up if these bite in practice.

use std::path::Path;

/// A chain whose mean post-burn-in log-likelihood is this many standard
/// deviations from the cross-chain grand mean is flagged as an outlier. 2.0 is
/// the ~95% band under a normal approximation: a chain beyond it sat in a
/// distinctly different part of the likelihood surface than its siblings — the
/// flat-ridge / side-mode signature. Named (not inlined) so the gate and the
/// summary share one threshold.
pub const CHAIN_LOGLIK_OUTLIER_Z: f64 = 2.0;

/// One chain's mean log-likelihood and its z-score against the between-chain
/// spread.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainZScore {
    /// 1-based chain number for display (`chain 1` … `chain N`), matching the
    /// user-facing convention used everywhere else in fit output.
    pub chain: usize,
    /// Mean log-likelihood over this chain's post-burn-in draws.
    pub mean_loglik: f64,
    /// `(mean_loglik − grand_mean) / SD(chain means)`. `0.0` when there is only
    /// one chain or every chain agrees (SD == 0).
    pub z: f64,
    /// `|z| > CHAIN_LOGLIK_OUTLIER_Z`.
    pub is_outlier: bool,
}

/// Per-chain z-scores from per-chain mean logliks, in chain order (index 0 =
/// chain 1). The centre and scale are the mean and *population* SD (n divisor)
/// of the N chain means themselves — the between-chain spread. A single chain,
/// or perfect agreement (SD ≈ 0), yields all-zero z-scores and nothing flagged
/// — the negative-control path. See the module docs for the small-N ceiling and
/// masking caveats.
pub fn chain_loglik_zscores(means: &[f64]) -> Vec<ChainZScore> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    let grand_mean = means.iter().sum::<f64>() / n as f64;
    // Population SD (n divisor): the spread of the actual set of chain means.
    // A single chain → treat spread as zero, so no chain can be an outlier
    // relative to a set of one.
    let sd = if n < 2 {
        0.0
    } else {
        let var = means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        var.sqrt()
    };
    means
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let z = if sd > 0.0 { (m - grand_mean) / sd } else { 0.0 };
            ChainZScore {
                chain: i + 1,
                mean_loglik: m,
                z,
                is_outlier: sd > 0.0 && z.abs() > CHAIN_LOGLIK_OUTLIER_Z,
            }
        })
        .collect()
}

/// The flagged chains as `"chain N"` labels, worst-|z| first. Empty when no
/// chain exceeds the threshold.
pub fn outlier_labels(scores: &[ChainZScore]) -> Vec<String> {
    let mut flagged: Vec<&ChainZScore> = scores.iter().filter(|s| s.is_outlier).collect();
    flagged.sort_by(|a, b| {
        b.z.abs()
            .partial_cmp(&a.z.abs())
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
    fn zscore_flags_a_single_clear_outlier() {
        // Five well-mixed chains near -100, one stuck far below at -400.
        let means = vec![-100.0, -101.0, -99.0, -100.5, -100.2, -400.0];
        let scores = chain_loglik_zscores(&means);
        assert_eq!(scores.len(), 6);
        // The stuck chain (index 5 → "chain 6") is the outlier.
        let stuck = &scores[5];
        assert_eq!(stuck.chain, 6);
        assert!(stuck.is_outlier, "stuck chain must flag: z = {}", stuck.z);
        assert!(stuck.z < -CHAIN_LOGLIK_OUTLIER_Z, "z must be strongly negative: {}", stuck.z);
        // The well-mixed chains must NOT flag.
        for sc in &scores[..5] {
            assert!(!sc.is_outlier, "well-mixed chain {} wrongly flagged (z={})", sc.chain, sc.z);
        }
        assert_eq!(outlier_labels(&scores), vec!["chain 6"]);
    }

    #[test]
    fn zscore_flags_nothing_when_well_mixed() {
        // Negative control: all chains agree — no outlier, and the z-scores are
        // small and finite (not a vacuous "everything is NaN" pass).
        let means = vec![-100.0, -100.3, -99.8, -100.1, -99.9, -100.2];
        let scores = chain_loglik_zscores(&means);
        assert!(scores.iter().all(|s| !s.is_outlier), "well-mixed set must flag nothing");
        assert!(scores.iter().all(|s| s.z.is_finite()), "z-scores must be finite");
        assert!(scores.iter().all(|s| s.z.abs() < CHAIN_LOGLIK_OUTLIER_Z));
        assert!(outlier_labels(&scores).is_empty());
    }

    #[test]
    fn single_chain_has_no_spread() {
        let scores = chain_loglik_zscores(&[-42.0]);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].z, 0.0);
        assert!(!scores[0].is_outlier);
    }

    #[test]
    fn identical_chains_yield_zero_z_not_nan() {
        // SD == 0 must not produce NaN z-scores (division-by-zero guard).
        let scores = chain_loglik_zscores(&[-10.0, -10.0, -10.0]);
        assert!(scores.iter().all(|s| s.z == 0.0), "identical chains → z = 0, not NaN");
        assert!(scores.iter().all(|s| !s.is_outlier));
    }

    /// End-to-end reader: synthesize a six-chain stage dir (`chain_N/trace.tsv`,
    /// one stuck) + a `draws.tsv` manifest, with a warm-up prefix in each trace
    /// that MUST be stripped. Six chains so the classic z clears the 2.0
    /// threshold for a single outlier (ceiling √(N−1) = √5 ≈ 2.24).
    #[test]
    fn reader_strips_warmup_and_names_the_stuck_chain() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_reader");
        std::fs::create_dir_all(&dir).unwrap();

        // Five good chains at ll ≈ -50, one stuck at ll ≈ -300. Each trace has 2
        // warm-up rows (ll ≈ -900) excluded by the last-K_c-rows rule (K_c = 3).
        let warmup = "1\t-900.0\t-905.0\n2\t-880.0\t-885.0\n";
        // dir numbering starts at 1 (pmmh/pgas convention).
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            let body = format!("step\tlog_likelihood\tlog_posterior\n{warmup}{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        for c in 1..=5 {
            write_trace(c, "3\t-50.0\t-52.0\n4\t-51.0\t-53.0\n5\t-49.0\t-51.0\n");
        }
        write_trace(6, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // stuck

        // draws.tsv: 3 kept draws per chain (chain ids 0..5), one param col.
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let means = read_chain_mean_logliks(&dir).expect("chain traces present");
        assert_eq!(means.len(), 6);
        // Warm-up (-900/-880) excluded → good chains ≈ -50, not dragged toward -890.
        for (i, m) in means.iter().take(5).enumerate() {
            assert!((m - (-50.0)).abs() < 1.0, "chain {} mean {} should be ≈ -50 (warm-up stripped)", i + 1, m);
        }
        assert!((means[5] - (-300.0)).abs() < 1.0, "chain 6 (stuck) mean {} should be ≈ -300", means[5]);

        let scores = chain_loglik_zscores(&means);
        assert!(scores[5].is_outlier, "stuck chain 6 must be flagged: z={}", scores[5].z);
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
