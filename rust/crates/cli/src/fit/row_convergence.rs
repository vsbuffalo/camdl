//! Per-row convergence of a *reported* quantity — R̂ and ESS of the value in
//! one artifact row, reduced over the draws that built that row, grouped by the
//! chain each draw came from (gh#794).
//!
//! This is a different question from the fit's parameter R̂, and the two come
//! apart in both directions. A reportable quantity can be far better determined
//! than the parameters behind it (`f_cfr` at split-R̂ 1.31 while `p_fatal` and
//! `h_care_surv` sit at 2.85 and 2.56), and a forecast can be undetermined while
//! every parameter looks settled. The fit's number is provenance; this one
//! describes the row.
//!
//! # Reduce over the MEAN, not the draw
//!
//! For a predictive row there are two candidate reductions and they do not
//! agree. Reducing over the posterior-predictive **draws** puts the observation
//! noise into the within-chain variance, so the denominator swamps the
//! between-chain numerator and R̂ is pulled toward 1 however much the chains
//! disagree. Measured on a five-chain Ebola fit (`cases_national`,
//! negative-binomial dispersion ≈ 5):
//!
//! ```text
//!                       chain medians        spread   mean within-chain 90% width
//! last observed day  53, 53, 66, 78, 116         62                           213
//! forecast +56 days  93, 125, 140, 155, 372     280                           762
//! ```
//!
//! The chains disagree fourfold about the eight-week forecast — 93 against 372
//! cases per day is a different epidemic, not a wide interval on one — and an R̂
//! over those draws sits near 1 and reports it sound. The dilution grows with
//! the observation dispersion, so it is worst exactly where mechanistic models
//! live. Both reductions are emitted and named apart; the mean one is the one to
//! act on.
//!
//! # The estimator
//!
//! [`sim::inference::convergence::rank_convergence`] — the rank-normalized split
//! R̂ and bulk-ESS of Vehtari et al. (2021) that `fit summary` already reports
//! for parameters (gh#84). One implementation, so a per-row number and a
//! per-parameter number mean the same thing.

use sim::inference::convergence::{rank_convergence, ConvergenceError};

/// R̂ and bulk-ESS of one artifact row's value across the chains that produced
/// it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowConvergence {
    /// Rank-normalized split R̂: `max(bulk, folded)`, the same headline
    /// `fit summary` prints per parameter.
    pub rhat: f64,
    /// Bulk effective sample size, pooled across chains.
    pub ess: f64,
}

impl RowConvergence {
    /// The R̂ cell. Empty when the reduction was refused (fewer than two chains,
    /// fewer than four draws per chain, a constant column) — never a bare `NaN`,
    /// which reads as a numerical failure rather than "not assessable here".
    pub fn rhat_cell(this: Option<&Self>) -> String {
        match this {
            Some(c) if c.rhat.is_finite() => format!("{:.4}", c.rhat),
            _ => String::new(),
        }
    }

    /// The ESS cell, rendered as an effective count. Empty on a refusal or a
    /// non-finite ESS.
    pub fn ess_cell(this: Option<&Self>) -> String {
        match this {
            Some(c) if c.ess.is_finite() => {
                if c.ess.fract() == 0.0 {
                    format!("{}", c.ess as i64)
                } else {
                    format!("{:.1}", c.ess)
                }
            }
            _ => String::new(),
        }
    }
}

/// The chain each replayed draw came from, in the order the row values were
/// accumulated.
///
/// Carried as an explicit partition rather than re-derived at each reduction
/// site: the values are pushed one per merged design cell, so the alignment
/// between a value and its chain is positional, and re-deriving it from the
/// subsample stride at a second site is how the two silently drift apart. A
/// `None` entry is a draw whose `draws.tsv` row carried no chain key — the
/// whole partition is then unusable and every reduction refuses.
#[derive(Debug, Clone, Default)]
pub struct ChainOfDraw(pub Vec<Option<usize>>);

impl ChainOfDraw {
    /// The chain ids as a dense slice, or `None` when any draw is unkeyed.
    fn dense(&self) -> Option<Vec<usize>> {
        self.0.iter().copied().collect()
    }
}

/// Split `values` (one per draw, in accumulation order) into one list per
/// chain, keyed by chain id and preserving each chain's draw order. Every draw
/// is kept — this is the grouping a per-chain BAND reduces, where an unequal
/// count is not a problem.
///
/// `None` when the cloud carries no chain keys, or when the key list is not 1:1
/// with the values.
pub fn group_by_chain(
    values: &[f64],
    chains: &ChainOfDraw,
) -> Option<std::collections::BTreeMap<usize, Vec<f64>>> {
    let ids = chains.dense()?;
    if ids.len() != values.len() {
        return None;
    }
    let mut by_chain: std::collections::BTreeMap<usize, Vec<f64>> =
        std::collections::BTreeMap::new();
    for (v, c) in values.iter().zip(ids.iter()) {
        by_chain.entry(*c).or_default().push(*v);
    }
    Some(by_chain)
}

/// [`group_by_chain`], narrowed to what R̂ will accept: at least two chains,
/// each truncated to their common length.
///
/// The truncation is what makes the input admissible: `rank_convergence`'s
/// between-chain variance uses one draw count for every chain, and a strided
/// posterior subsample hands out counts that differ by one. Each chain keeps its
/// FIRST `m` draws, so the retained block stays contiguous in sweep order and
/// the split-half comparison still compares a chain's own two halves.
///
/// Returns `None` when the partition is unusable — an unkeyed cloud, or fewer
/// than two chains with any draws.
pub fn partition_by_chain(values: &[f64], chains: &ChainOfDraw) -> Option<Vec<Vec<f64>>> {
    let by_chain = group_by_chain(values, chains)?;
    if by_chain.len() < 2 {
        return None;
    }
    let m = by_chain.values().map(|v| v.len()).min().unwrap_or(0);
    if m == 0 {
        return None;
    }
    Some(by_chain.into_values().map(|mut v| { v.truncate(m); v }).collect())
}

/// R̂ and bulk-ESS of one row's per-draw values, grouped by chain.
///
/// `None` when the reduction is not available or was refused: an unkeyed cloud,
/// one chain, fewer than four retained draws per chain, a non-finite value, or a
/// column that never moved. Every one of those is "not assessable", which is
/// rendered as an empty cell — distinct from a computed number, and never a
/// silent 1.0.
pub fn row_convergence(values: &[f64], chains: &ChainOfDraw) -> Option<RowConvergence> {
    let grouped = partition_by_chain(values, chains)?;
    reduce(&grouped)
}

/// The reduction itself, over an already-grouped `chains[c][i]`. Split out so a
/// test can hand it an exact partition and assert against a hand-computable
/// case.
pub fn reduce(grouped: &[Vec<f64>]) -> Option<RowConvergence> {
    match rank_convergence(grouped) {
        Ok(rc) => Some(RowConvergence { rhat: rc.rhat, ess: rc.ess_bulk }),
        // Every variant is a property of the input, not a numerical failure:
        // the honest report is an empty cell.
        Err(
            ConvergenceError::TooFewChains { .. }
            | ConvergenceError::TooFewDraws { .. }
            | ConvergenceError::UnequalChainLengths { .. }
            | ConvergenceError::NonFiniteDraw { .. }
            | ConvergenceError::ConstantDraws { .. },
        ) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(ids: &[usize]) -> ChainOfDraw {
        ChainOfDraw(ids.iter().map(|c| Some(*c)).collect())
    }

    #[test]
    fn partition_groups_by_chain_and_truncates_to_the_common_length() {
        // Chain 0 has 5 draws, chain 1 has 4 — a strided subsample's shape.
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0];
        let ids = keys(&[0, 0, 0, 0, 0, 1, 1, 1, 1]);
        let g = partition_by_chain(&values, &ids).expect("two keyed chains");
        assert_eq!(g.len(), 2);
        // Each chain keeps its FIRST 4 — chain 0's trailing 5.0 is dropped.
        assert_eq!(g[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(g[1], vec![10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn partition_refuses_an_unkeyed_cloud_and_a_single_chain() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert!(
            partition_by_chain(&values, &ChainOfDraw(vec![Some(0), None, Some(1), Some(1)]))
                .is_none(),
            "a draws.tsv with no chain column cannot be grouped"
        );
        assert!(
            partition_by_chain(&values, &keys(&[0, 0, 0, 0])).is_none(),
            "R̂ compares chains; one chain has nothing to compare against"
        );
    }

    #[test]
    fn reduce_refuses_a_constant_row_rather_than_publishing_a_number() {
        let g = vec![vec![7.0; 8], vec![7.0; 8]];
        assert!(reduce(&g).is_none(), "a row that never moved has no R̂");
    }

    #[test]
    fn reduce_refuses_fewer_than_four_draws_per_chain() {
        let g = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert!(reduce(&g).is_none(), "split-R̂ needs at least 4 draws per chain");
    }

    /// Pin the per-row number to the shared estimator on a case whose answer is
    /// computable by hand, so the column cannot drift from the implementation it
    /// claims to use.
    ///
    /// Two chains of eight draws, chain 1 = chain 0 + 100. Every value is
    /// distinct, so the rank transform maps the 16 pooled draws onto the fixed
    /// Blom scores `Φ⁻¹((r − 3/8)/(16 + 1/4))`, and the split halves are the four
    /// LOWEST scores (chain 0) against the four HIGHEST (chain 1). Separated
    /// chains → a large R̂; the assertion is against the same function
    /// `fit summary` calls, evaluated here on the identical input.
    #[test]
    fn per_row_rhat_is_the_shared_rank_normalized_estimator() {
        let a: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..8).map(|i| 100.0 + i as f64).collect();
        let grouped = vec![a, b];
        let got = reduce(&grouped).expect("two chains of eight draws are assessable");
        let oracle = sim::inference::convergence::rank_convergence(&grouped)
            .expect("the shared estimator accepts this input");
        assert_eq!(got.rhat, oracle.rhat, "rhat must be the shared headline R̂");
        assert_eq!(got.ess, oracle.ess_bulk, "ess must be the shared bulk-ESS");
        assert!(
            got.rhat > 2.0,
            "fully separated chains are a large R̂, got {}",
            got.rhat
        );
    }

    #[test]
    fn cells_are_empty_on_a_refusal_and_formatted_otherwise() {
        assert_eq!(RowConvergence::rhat_cell(None), "");
        assert_eq!(RowConvergence::ess_cell(None), "");
        let c = RowConvergence { rhat: 1.0234567, ess: 412.0 };
        assert_eq!(RowConvergence::rhat_cell(Some(&c)), "1.0235");
        assert_eq!(RowConvergence::ess_cell(Some(&c)), "412");
        let c = RowConvergence { rhat: f64::NAN, ess: 5.83 };
        assert_eq!(RowConvergence::rhat_cell(Some(&c)), "");
        assert_eq!(RowConvergence::ess_cell(Some(&c)), "5.8");
    }
}
