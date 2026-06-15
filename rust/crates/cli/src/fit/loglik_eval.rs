//! Clean re-evaluation of IF2 final-iteration parameter means.
//!
//! After IF2 finishes, each chain's final-iteration parameter means —
//! the IF2 estimator coef(mif2_out) in pomp's terminology — is
//! re-scored with `M` independent high-particle particle-filter
//! replicates and combined via logmeanexp on the likelihood scale.
//! The delta-method standard error of the combined estimator is
//! reported alongside.
//!
//! This matches pomp's documented post-mif2 workflow (King, Ionides
//! & Bretó; Ionides, Nguyen, Atchadé, Stoev & King 2015 PNAS,
//! supplementary materials; pomp manual on `logmeanexp` and
//! `pfilter`). The point estimate is the IF2 final-iteration mean;
//! the loglik is reported by re-evaluating with high-particle clean
//! PF replicates rather than reusing the noisy in-run PF logliks
//! that drove the IF2 perturbation.
//!
//! Why this is needed (and what it replaces). The IF2 algorithm
//! uses few-particle in-run PFs (~500 particles) as its weighting
//! target. Reporting the iteration-argmax of those noisy in-run
//! logliks as "the MLE loglik" is selection-biased — argmax over a
//! sample of noisy estimates is upward-biased relative to the
//! population mean. Re-scoring at the final-iteration mean with
//! high-particle replicates removes that bias while keeping the
//! IF2 algorithm's theoretical anchor (Ionides et al. 2015 prove
//! convergence to the MLE in the final-iter mean).
//!
//! Module split (kept from earlier design):
//! - `run_loglik_eval_with_scorer` is generic over the scorer
//!   closure so tests can inject a deterministic synthetic scorer
//!   without paying for real PF calls.
//! - `run_loglik_eval` is the production entry point that wires in
//!   `runner::run_quick_pfilter_full`.
//!
//! See `docs/dev/notes/2026-04-27-clean-eval-strip.md` for the
//! rationale behind stripping the prior 3-candidate construction
//! (uncited; introduced selection bias on top of the bias it was
//! meant to fix). The earlier proposal at
//! `docs/dev/proposals/2026-04-24-if2-scout-findings-remediation.md`
//! is preserved as the original design record but no longer
//! reflects the implementation.

use crate::evidence;
use crate::fit::config_v2::{LoglikEvalConfig, CombineMode};
use crate::fit::runner::{self, FitRunConfig};
use sim::inference::if2::IF2Result;

// ──────────────────────────────────────────────────────────────────────
// Filter-health summary
// ──────────────────────────────────────────────────────────────────────

/// Filter-health summary from one PF run. Aggregates `ess_trace`
/// (per-observation ESS) and `ll_increments` (per-observation
/// log-likelihood increment) into the four numbers users care about
/// for "is the filter healthy at this θ?". Surfaces in
/// `chain_evaluations.tsv` per chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterStats {
    /// Mean ESS across all observation steps. NaN when no observations.
    pub ess_mean: f64,
    /// Min ESS across all observation steps. NaN when no observations.
    pub ess_min: f64,
    /// Observation index (0-based) where ESS is worst. `None` when no
    /// observations.
    pub ess_min_step: Option<usize>,
    /// Count of observation steps where ll_increment is −∞ — the
    /// "filter completely lost the data at this step" signal. Different
    /// from low ESS: −∞ means *zero* particles agreed with the
    /// observation; low ESS means few particles dominated the weight.
    pub n_neg_inf_increments: usize,
    /// First observation step where ll_increment is −∞, or `None` if
    /// there are no such steps. Useful for localising model
    /// mis-specification — the `data[i]` you should look at.
    pub neg_inf_first_step: Option<usize>,
}

impl FilterStats {
    /// Sentinel value for failed PF runs. ESS fields are NaN, counts
    /// are zero. Distinguishable from "successful PF with no
    /// observations" because in that case `ess_min_step` is also `None`
    /// — but the (NaN, NaN) ESS pair signals failure to a reader.
    pub fn failed() -> Self {
        Self {
            ess_mean: f64::NAN,
            ess_min:  f64::NAN,
            ess_min_step: None,
            n_neg_inf_increments: 0,
            neg_inf_first_step:   None,
        }
    }

    /// Aggregate per-observation ess + ll_increment series into the
    /// summary numbers. Pure on its inputs; testable without running
    /// a real PF. `ess_trace` and `ll_increments` come straight from
    /// `PFilterResult`.
    pub fn from_pfilter_result(ess_trace: &[f64], ll_increments: &[f64]) -> Self {
        if ess_trace.is_empty() {
            return Self {
                ess_mean: f64::NAN,
                ess_min:  f64::NAN,
                ess_min_step: None,
                n_neg_inf_increments: ll_increments.iter()
                    .filter(|x| !x.is_finite() && x.is_sign_negative()).count(),
                neg_inf_first_step: ll_increments.iter().position(
                    |x| !x.is_finite() && x.is_sign_negative()),
            };
        }
        let n = ess_trace.len() as f64;
        let ess_mean = ess_trace.iter().sum::<f64>() / n;
        let (ess_min_step, ess_min) = ess_trace.iter().enumerate()
            .fold((0usize, f64::INFINITY), |(best_i, best_v), (i, &v)| {
                if v < best_v { (i, v) } else { (best_i, best_v) }
            });
        let n_neg_inf = ll_increments.iter()
            .filter(|x| !x.is_finite() && x.is_sign_negative()).count();
        let neg_inf_first = ll_increments.iter().position(
            |x| !x.is_finite() && x.is_sign_negative());
        Self {
            ess_mean,
            ess_min,
            ess_min_step: Some(ess_min_step),
            n_neg_inf_increments: n_neg_inf,
            neg_inf_first_step: neg_inf_first,
        }
    }

    /// Combine M independent replicates into a single representative
    /// `FilterStats`. Mean of mean-ESS, min of min-ESS (worst case),
    /// step from the worst replicate's worst step. Sums neg-inf
    /// counts. Used by clean-eval to summarise across the M PF
    /// replicates per chain.
    pub fn aggregate(replicates: &[Self]) -> Self {
        if replicates.is_empty() { return Self::failed(); }
        let n = replicates.len() as f64;
        let mean_of_means: f64 = replicates.iter().map(|s| s.ess_mean).sum::<f64>() / n;
        let (min_step, min_v) = replicates.iter()
            .filter_map(|s| s.ess_min_step.map(|i| (i, s.ess_min)))
            .fold((None, f64::INFINITY), |(best_i, best_v), (i, v)| {
                if v < best_v { (Some(i), v) } else { (best_i, best_v) }
            });
        let n_neg_inf: usize = replicates.iter().map(|s| s.n_neg_inf_increments).sum();
        let first_neg_inf = replicates.iter()
            .filter_map(|s| s.neg_inf_first_step)
            .min();
        Self {
            ess_mean: mean_of_means,
            ess_min:  min_v,
            ess_min_step: min_step,
            n_neg_inf_increments: n_neg_inf,
            neg_inf_first_step:   first_neg_inf,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Per-chain clean-eval score
// ──────────────────────────────────────────────────────────────────────

/// One chain's clean-evaluation result. The chain's IF2
/// final-iteration parameter means are scored with `M` independent
/// high-particle PF replicates; `loglik` and `se` summarise the
/// combined estimator across replicates, and `filter_stats`
/// aggregates filter health across the same replicates.
#[derive(Debug, Clone)]
pub struct ChainScore {
    pub chain_id: usize,
    /// IF2 final-iteration parameter means — the chain's MLE estimate
    /// in the IF2 framework (Ionides et al. 2015).
    pub theta: Vec<f64>,
    /// Combined log-likelihood across the per-replicate logliks.
    /// `logmeanexp` by default (unbiased on the likelihood scale,
    /// matches pomp); `mean` (arithmetic mean of logs) when configured.
    pub loglik: f64,
    /// Standard error of the combined estimator. For `LogMeanExp`,
    /// the delta-method SE on the log scale (`SD(L_i) / (L̄ √M)`,
    /// matches pomp's `pfilter` reporting). For `Mean`, the standard
    /// arithmetic-mean SE (`sd(per_rep) / √M`).
    pub se: f64,
    /// Filter-health summary aggregated over the M replicates.
    pub filter_stats: FilterStats,
}

/// Per-chain clean-evaluation results plus an index into `per_chain`
/// naming the highest-loglik chain. Consumers wanting the overall
/// winner read `per_chain[overall_winner_idx]`; consumers wanting
/// per-chain detail (the gate's chain-spread leg) iterate `per_chain`.
#[derive(Debug, Clone)]
pub struct LoglikEvalOutcome {
    pub per_chain: Vec<ChainScore>,
    pub overall_winner_idx: usize,
}

// ──────────────────────────────────────────────────────────────────────
// Production entry point
// ──────────────────────────────────────────────────────────────────────

/// Production entry point: clean-evaluate every chain's IF2
/// final-iteration mean using `runner::run_quick_pfilter_full` as the
/// scorer (returns loglik + filter stats so we can surface ESS-at-θ̂).
pub fn run_loglik_eval(
    run_config: &FitRunConfig,
    results: &[(usize, IF2Result)],
    cfg: &LoglikEvalConfig,
    seed: u64,
) -> Result<LoglikEvalOutcome, String> {
    run_loglik_eval_with_scorer(
        results,
        cfg,
        seed,
        |theta, n_particles, pf_seed| {
            // gh#224: a ruled-out θ scores −∞; a structural error (model/config
            // can't run) aborts the clean-eval rather than reporting a
            // degenerate loglik at θ̂.
            runner::ruled_out_or_surface(
                runner::run_quick_pfilter_full(run_config, theta, n_particles, pf_seed))
                .map_err(|e| format!("loglik-eval: structural error at θ̂: {}", e))
        },
    )
}

/// Test-friendly inner core. `scorer(theta, n_particles, seed) ->
/// (ll, FilterStats)` is called `n_chains × n_replicates` times.
///
/// Seed scheme: `pf_seed = seed + chain_id*10_000 + rep_k`. Replicate
/// counts above ~10_000 would alias into the next chain's seed range;
/// in practice `n_replicates` is a small constant (default M=8) and
/// `chain_id` is single-digit, so collisions don't occur. If either
/// scale changes, revisit the offsets.
pub fn run_loglik_eval_with_scorer<F>(
    results: &[(usize, IF2Result)],
    cfg: &LoglikEvalConfig,
    seed: u64,
    scorer: F,
) -> Result<LoglikEvalOutcome, String>
where
    F: Fn(&[f64], usize, u64) -> Result<(f64, FilterStats), String>,
{
    if results.is_empty() {
        return Err("run_loglik_eval: no chain results to score".into());
    }
    if cfg.n_replicates == 0 {
        return Err("run_loglik_eval: n_replicates must be ≥ 1".into());
    }
    if cfg.n_particles == 0 {
        return Err("run_loglik_eval: n_particles must be ≥ 1".into());
    }

    let mut per_chain: Vec<ChainScore> = Vec::with_capacity(results.len());

    for (chain_id, if2) in results.iter() {
        if if2.iterations.is_empty() {
            return Err(format!(
                "run_loglik_eval: chain {} has no IF2 iterations", chain_id));
        }
        // The IF2 estimator: final-iteration parameter means.
        // Matches pomp's coef(mif2_out).
        let theta = if2.iterations.last().unwrap().param_means.clone();

        let mut per_rep = Vec::with_capacity(cfg.n_replicates);
        let mut per_rep_stats = Vec::with_capacity(cfg.n_replicates);
        for k in 0..cfg.n_replicates {
            let pf_seed = seed
                .wrapping_add((*chain_id as u64).wrapping_mul(10_000))
                .wrapping_add(k as u64);
            let (ll, stats) = scorer(&theta, cfg.n_particles, pf_seed)?;
            per_rep.push(ll);
            per_rep_stats.push(stats);
        }
        let (loglik, se) = combine_with_se(&per_rep, cfg.combine);
        let filter_stats = FilterStats::aggregate(&per_rep_stats);

        per_chain.push(ChainScore {
            chain_id: *chain_id,
            theta,
            loglik,
            se,
            filter_stats,
        });
    }

    // Overall winner: argmax of per-chain logliks. NaN-safe — NaN never
    // wins; if every chain produced NaN, error explicitly rather than
    // silently picking index 0.
    let overall_winner_idx = per_chain.iter().enumerate()
        .filter(|(_, s)| !s.loglik.is_nan())
        .fold(None::<usize>, |best, (i, s)| match best {
            None => Some(i),
            Some(prev) if s.loglik > per_chain[prev].loglik => Some(i),
            Some(prev) => Some(prev),
        })
        .ok_or_else(|| "run_loglik_eval: every chain produced NaN loglik".to_string())?;

    Ok(LoglikEvalOutcome { per_chain, overall_winner_idx })
}

/// Combine `M` per-replicate logliks into a (combined, SE) pair. The
/// SE formula matches the combine mode — `logmeanexp_with_se` for
/// `LogMeanExp`, `sample_sd / √M` for `Mean` — so the reported SE is
/// always the SE of the reported point estimate.
pub fn combine_with_se(xs: &[f64], mode: CombineMode) -> (f64, f64) {
    match mode {
        CombineMode::LogMeanExp => evidence::logmeanexp_with_se(xs),
        CombineMode::Mean => {
            if xs.is_empty() {
                return (f64::NEG_INFINITY, f64::NAN);
            }
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let se = evidence::sample_sd(xs) / (xs.len() as f64).sqrt();
            (mean, se)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::if2::{IF2IterResult, IF2Result};

    fn iter(n: usize, loglik: f64, perturbed: f64, params: Vec<f64>) -> IF2IterResult {
        IF2IterResult {
            iteration: n,
            loglik,
            if2_perturbed_loglik: perturbed,
            param_means: params,
            param_diag: vec![],
        }
    }

    fn synthetic_result(iters: Vec<IF2IterResult>) -> IF2Result {
        IF2Result {
            mle: iters.last().map(|it| it.param_means.clone()).unwrap_or_default(),
            final_loglik: iters.last().map(|it| it.loglik).unwrap_or(f64::NAN),
            last_loglik: iters.last().map(|it| it.if2_perturbed_loglik).unwrap_or(f64::NAN),
            iterations: iters,
        }
    }

    #[test]
    fn run_loglik_eval_uses_final_iteration_param_means() {
        // The chain's reported θ̂ must come from iterations.last().
        // Even if earlier iterations had different param_means, the
        // final iter is the one re-scored — pomp convention.
        let r = synthetic_result(vec![
            iter(0, -10.0, -10.0, vec![1.0, 2.0]),
            iter(1, -8.0,  -8.0,  vec![3.0, 4.0]),
            iter(2, -5.0,  -5.0,  vec![5.0, 6.0]), // final
        ]);
        let results = vec![(0usize, r)];

        // Scorer just records the theta it was called with.
        let scorer = |theta: &[f64], _: usize, _: u64| {
            // Return the sum of theta as the loglik so we can verify.
            Ok((theta.iter().sum::<f64>(), FilterStats::failed()))
        };
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 1, combine: CombineMode::Mean,
        };
        let out = run_loglik_eval_with_scorer(&results, &cfg, 0, scorer).unwrap();

        assert_eq!(out.per_chain.len(), 1);
        assert_eq!(out.per_chain[0].theta, vec![5.0, 6.0],
            "theta must be iterations.last().param_means");
        // Sum of [5.0, 6.0] = 11.0, returned by scorer.
        assert!((out.per_chain[0].loglik - 11.0).abs() < 1e-12);
    }

    #[test]
    fn run_loglik_eval_argmax_picks_higher_clean_loglik() {
        // Two chains, identical IF2Result structure but the
        // deterministic scorer prefers chain-1's thetas. Clean-eval
        // picks chain 1 regardless of in-run numbers.
        let mk_chain = |a: f64| synthetic_result(vec![
            iter(0, -1000.0, -1000.0, vec![a, a + 1.0]),
        ]);
        let results = vec![
            (0usize, mk_chain(0.0)),  // small thetas
            (1usize, mk_chain(10.0)), // large thetas
        ];
        // Scorer: -10 for small thetas, -5 for large thetas.
        let scorer = |theta: &[f64], _: usize, _: u64| {
            Ok((if theta[0] < 5.0 { -10.0 } else { -5.0 }, FilterStats::failed()))
        };
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 4, combine: CombineMode::LogMeanExp,
        };
        let out = run_loglik_eval_with_scorer(&results, &cfg, 42, scorer).unwrap();

        assert_eq!(out.per_chain.len(), 2);
        assert!((out.per_chain[0].loglik - (-10.0)).abs() < 1e-12);
        assert!((out.per_chain[1].loglik - (-5.0)).abs() < 1e-12);
        assert_eq!(out.overall_winner_idx, 1,
            "overall winner is the chain with the higher clean loglik");
        // Constant scorer → no per-rep spread → SE = 0.
        for s in &out.per_chain {
            assert!(s.se.abs() < 1e-12, "constant scorer should give SE = 0, got {}", s.se);
        }
    }

    #[test]
    fn run_loglik_eval_se_for_mean_combine_matches_sd_over_sqrt_m() {
        // Mean combine: SE = sample_sd / √M. Verify on a known input.
        let r = synthetic_result(vec![iter(0, -5.0, -5.0, vec![1.0])]);
        let results = vec![(7usize, r)];

        // M = 4 replicates; values 1, 2, 3, 4 (mapped from rep_k via
        // the seed scheme: pf_seed = 0 + 7*10_000 + k → rep_k = seed % 10_000).
        let scorer = |_t: &[f64], _: usize, seed: u64| {
            let rep_k = (seed % 10_000) as f64;
            Ok((rep_k + 1.0, FilterStats::failed()))
        };
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 4, combine: CombineMode::Mean,
        };
        let out = run_loglik_eval_with_scorer(&results, &cfg, 0, scorer).unwrap();

        // sd of [1, 2, 3, 4] = sqrt(((1+4+9+16) - 100/4) / 3) = sqrt(5/3)·...
        // sample_sd uses N-1 denom: var = sum((x - mean)^2) / 3 = 1.25·...
        // Actually let's compute: mean=2.5, sq devs = [2.25, 0.25, 0.25, 2.25]
        // sum = 5, var = 5/3, sd = sqrt(5/3) ≈ 1.2910.
        // SE = sd/√M = 1.2910 / 2 ≈ 0.6455.
        let mean = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
        let var = ((1.0_f64 - mean).powi(2) + (2.0_f64 - mean).powi(2)
                 + (3.0_f64 - mean).powi(2) + (4.0_f64 - mean).powi(2)) / 3.0;
        let expected_se = var.sqrt() / 2.0;
        assert!((out.per_chain[0].se - expected_se).abs() < 1e-9,
            "Mean-combine SE should match sd/√M; expected {}, got {}",
            expected_se, out.per_chain[0].se);
    }

    #[test]
    fn run_loglik_eval_se_for_logmeanexp_combine_matches_delta_method() {
        // LogMeanExp combine: SE = SD(L_i) / (L̄ √M). Matches the
        // delta-method formula. Synthetic deterministic per-rep
        // logliks make the answer analytical.
        let r = synthetic_result(vec![iter(0, -5.0, -5.0, vec![1.0])]);
        let results = vec![(0usize, r)];
        // Two replicates: -100 and -98.
        let scorer = |_t: &[f64], _: usize, seed: u64| {
            let rep_k = (seed % 10_000) as i64;
            Ok(([-100.0_f64, -98.0_f64][rep_k as usize], FilterStats::failed()))
        };
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 2, combine: CombineMode::LogMeanExp,
        };
        let out = run_loglik_eval_with_scorer(&results, &cfg, 0, scorer).unwrap();

        // Verify against evidence::logmeanexp_with_se directly.
        let (expected_ll, expected_se) = evidence::logmeanexp_with_se(&[-100.0, -98.0]);
        assert!((out.per_chain[0].loglik - expected_ll).abs() < 1e-9);
        assert!((out.per_chain[0].se - expected_se).abs() < 1e-9);
    }

    #[test]
    fn run_loglik_eval_combine_mean_vs_logmeanexp_differ() {
        // For non-degenerate replicates, mean(ℓ) < logmeanexp(ℓ).
        let r = synthetic_result(vec![iter(0, -5.0, -5.0, vec![0.0])]);
        let results = vec![(0usize, r)];
        let scorer = |_t: &[f64], _: usize, seed: u64| {
            let rep_k = (seed % 10_000) as f64;
            Ok((-10.0 + rep_k, FilterStats::failed()))
        };
        let cfg_mean = LoglikEvalConfig {
            n_particles: 1, n_replicates: 4, combine: CombineMode::Mean,
        };
        let cfg_lme = LoglikEvalConfig {
            n_particles: 1, n_replicates: 4, combine: CombineMode::LogMeanExp,
        };
        let mean_out = run_loglik_eval_with_scorer(&results, &cfg_mean, 0, scorer).unwrap();
        let lme_out  = run_loglik_eval_with_scorer(&results, &cfg_lme,  0, scorer).unwrap();
        // Same theta selected; the scores differ in the expected direction.
        assert!(lme_out.per_chain[0].loglik > mean_out.per_chain[0].loglik);
    }

    #[test]
    fn run_loglik_eval_errors_on_empty_iterations() {
        let r = synthetic_result(vec![]);
        let results = vec![(0usize, r)];
        let scorer = |_: &[f64], _: usize, _: u64| Ok((0.0, FilterStats::failed()));
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 1, combine: CombineMode::Mean,
        };
        assert!(run_loglik_eval_with_scorer(&results, &cfg, 0, scorer).is_err());
    }

    #[test]
    fn run_loglik_eval_errors_on_all_nan_chains() {
        // If every chain's combined loglik is NaN, error rather than
        // silently picking chain 0.
        let mk = || synthetic_result(vec![iter(0, -5.0, -5.0, vec![1.0])]);
        let results = vec![(0usize, mk()), (1usize, mk())];
        let scorer = |_: &[f64], _: usize, _: u64| Ok((f64::NAN, FilterStats::failed()));
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 2, combine: CombineMode::Mean,
        };
        let result = run_loglik_eval_with_scorer(&results, &cfg, 0, scorer);
        assert!(result.is_err(), "all-NaN chains must error explicitly");
        assert!(result.unwrap_err().contains("NaN"));
    }

    /// The empirical anchor the original module never had: across many
    /// synthetic trials, does clean-eval produce a loglik estimate that
    /// is more accurate (lower RMSE relative to the true loglik) than a
    /// single in-run-style PF call would be?
    ///
    /// Setup. We model two PF regimes by their estimator distributions:
    ///
    ///   - **In-run PF**: low particle count (~500), high per-rep
    ///     variance. Modelled as `log L̂_in = log L − σ²/2 + σ Z` with
    ///     σ ≈ 3 nats and Z ~ Normal(0,1). The −σ²/2 captures the
    ///     standard PF downward bias on the log scale (Pitt, Silva,
    ///     Giordani & Kohn 2012, JRSS-B; the PF estimator is unbiased
    ///     on the likelihood scale but downward-biased by σ²/2 on the
    ///     log scale).
    ///   - **Clean PF**: high particle count, low per-rep variance.
    ///     Modelled with σ ≈ 0.3 nats. Same downward-bias structure,
    ///     just an order of magnitude tighter.
    ///
    /// Across N_TRIALS trials, we compare:
    ///
    ///   - Clean-eval (M=8 replicates of clean PF combined via
    ///     `logmeanexp`): the post-strip workflow.
    ///   - Single in-run PF call: what we'd report if we just used
    ///     `IF2Result.final_loglik` directly without re-evaluation.
    ///
    /// Assertions: clean-eval has lower RMSE *and* lower mean absolute
    /// error relative to the true loglik. Both directions of the
    /// "clean is more accurate" claim are checked. The numerical
    /// margins are intentionally loose so the test is robust to seed
    /// variation while still catching a regression that would let the
    /// in-run estimator beat clean-eval.
    ///
    /// This test validates the post-strip clean-eval system's central
    /// claim. See `docs/dev/notes/2026-04-27-clean-eval-strip.md` for
    /// the rationale.
    #[test]
    fn clean_eval_reduces_bias_and_variance_relative_to_single_in_run_pf() {
        use sim::rng::StatefulRng;

        const TRUE_LL: f64 = -100.0;
        const SIGMA_IN_RUN: f64 = 3.0;
        const SIGMA_CLEAN: f64 = 0.3;
        const N_TRIALS: usize = 200;

        // Standard PF estimator on log scale: E[log L̂] = log L − σ²/2.
        let pf_sample = |rng: &mut StatefulRng, sigma: f64| -> f64 {
            TRUE_LL - sigma * sigma / 2.0 + sigma * rng.normal()
        };

        // IF2Result with one iteration. The θ doesn't matter — the
        // synthetic scorers don't read it.
        let r = synthetic_result(vec![iter(0, TRUE_LL, TRUE_LL, vec![1.0])]);
        let results = vec![(0usize, r)];
        let cfg = LoglikEvalConfig {
            n_particles: 1,
            n_replicates: 8,
            combine: CombineMode::LogMeanExp,
        };

        let mut clean_errs = Vec::with_capacity(N_TRIALS);
        let mut in_run_errs = Vec::with_capacity(N_TRIALS);

        for trial in 0..N_TRIALS {
            // Each trial gets a unique seed-namespace. Within the
            // trial, the clean-eval seed scheme produces unique seeds
            // per replicate via `pf_seed = base + chain*10_000 + rep_k`.
            let trial_seed = 1_000_000_u64 + (trial as u64).wrapping_mul(100);

            // Closure that mimics a high-particle clean PF.
            let clean_scorer = |_: &[f64], _: usize, seed: u64| {
                let mut rng = StatefulRng::new(seed);
                Ok((pf_sample(&mut rng, SIGMA_CLEAN), FilterStats::failed()))
            };
            let outcome = run_loglik_eval_with_scorer(
                &results, &cfg, trial_seed, clean_scorer,
            ).unwrap();
            let clean_est = outcome.per_chain[0].loglik;
            clean_errs.push((clean_est - TRUE_LL).abs());

            // A single in-run-style estimate at the same θ (one PF
            // call, low particle count) — what we'd report without
            // clean re-evaluation.
            let mut in_run_rng = StatefulRng::new(trial_seed.wrapping_add(999));
            let in_run_est = pf_sample(&mut in_run_rng, SIGMA_IN_RUN);
            in_run_errs.push((in_run_est - TRUE_LL).abs());
        }

        let rmse = |xs: &[f64]| -> f64 {
            (xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64).sqrt()
        };
        let mean = |xs: &[f64]| -> f64 {
            xs.iter().sum::<f64>() / xs.len() as f64
        };

        let clean_rmse = rmse(&clean_errs);
        let in_run_rmse = rmse(&in_run_errs);
        let clean_mean_err = mean(&clean_errs);
        let in_run_mean_err = mean(&in_run_errs);

        // Direct claim: clean-eval is more accurate.
        assert!(
            clean_rmse < in_run_rmse,
            "clean-eval RMSE ({:.3}) must be < single-in-run RMSE ({:.3}); \
             this is the bias-reduction claim",
            clean_rmse, in_run_rmse
        );
        assert!(
            clean_mean_err < in_run_mean_err,
            "clean-eval mean abs error ({:.3}) must be < single-in-run \
             mean abs error ({:.3})",
            clean_mean_err, in_run_mean_err
        );

        // Quantitative: with σ_clean=0.3 and σ_in_run=3.0, the
        // theoretical std-ratio favoring clean is 10× before averaging,
        // and `logmeanexp` over M=8 replicates further tightens clean.
        // Asserting at least 5× margin keeps the test robust to seed
        // variation while still catching a regression where clean-eval
        // becomes no better than a single in-run call.
        assert!(
            clean_rmse * 5.0 < in_run_rmse,
            "clean-eval RMSE ({:.3}) should be ≥5× lower than \
             single-in-run RMSE ({:.3}); observed ratio is {:.3}× — \
             the clean-eval pipeline is not delivering its expected \
             accuracy gain",
            clean_rmse, in_run_rmse, in_run_rmse / clean_rmse
        );
    }

    #[test]
    fn filter_stats_from_pfilter_result_basic() {
        let ess = vec![1000.0, 800.0, 200.0, 400.0, 950.0];
        let ll_incr = vec![-5.0, -6.0, -10.0, -7.0, -5.0];
        let s = FilterStats::from_pfilter_result(&ess, &ll_incr);
        assert!((s.ess_mean - 670.0).abs() < 1e-9, "mean: {}", s.ess_mean);
        assert!((s.ess_min - 200.0).abs() < 1e-9, "min: {}", s.ess_min);
        assert_eq!(s.ess_min_step, Some(2));
        assert_eq!(s.n_neg_inf_increments, 0);
        assert_eq!(s.neg_inf_first_step, None);
    }

    #[test]
    fn filter_stats_counts_neg_inf_increments() {
        let ess = vec![900.0, 800.0];
        let ll_incr = vec![-5.0, f64::NEG_INFINITY];
        let s = FilterStats::from_pfilter_result(&ess, &ll_incr);
        assert_eq!(s.n_neg_inf_increments, 1);
        assert_eq!(s.neg_inf_first_step, Some(1));
    }

    #[test]
    fn filter_stats_aggregate_min_is_worst_case_across_replicates() {
        let r1 = FilterStats {
            ess_mean: 1000.0, ess_min: 500.0, ess_min_step: Some(3),
            n_neg_inf_increments: 0, neg_inf_first_step: None,
        };
        let r2 = FilterStats {
            ess_mean: 800.0, ess_min: 100.0, ess_min_step: Some(7),
            n_neg_inf_increments: 2, neg_inf_first_step: Some(7),
        };
        let agg = FilterStats::aggregate(&[r1, r2]);
        assert!((agg.ess_mean - 900.0).abs() < 1e-9);
        assert!((agg.ess_min - 100.0).abs() < 1e-9);
        assert_eq!(agg.ess_min_step, Some(7));
        assert_eq!(agg.n_neg_inf_increments, 2);
        assert_eq!(agg.neg_inf_first_step, Some(7));
    }
}
