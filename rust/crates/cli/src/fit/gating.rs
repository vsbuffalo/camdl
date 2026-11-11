//! The convergence check at the one seam where a stored fit is consumed as a
//! warm start — `starts = { from_mle = "@handle" }` or
//! `{ from_posterior = "@handle" }` — and the compound chain-agreement gate
//! an IF2 run is judged by at its end.
//!
//! [`check_source_converged`] refuses a source whose stored verdict is not
//! converged unless `--allow-nonconverged-source` is passed: that is the
//! purpose the old pre-refine gate served, kept where the upstream is
//! consumed rather than inside a `[stages]` pipeline (proposal
//! 2026-09-08-workflow-first-fit-config, §8 item 11). The post-refine
//! regression gate it used to pair with guarded an in-process handoff that no
//! longer exists; across invocations a regression is visible in `fit table`.
//!
//! [`check_scout_convergence`] is the compound gate itself — the tail
//! chain-agreement Â on every structural parameter (every one not declared
//! `perturb_only_at_t0`) below `gate.a_thresh`, and the inter-chain clean-eval
//! spread below its decibans floor — which `fit summary` renders for every
//! IF2 run and the source check applies to an IF2 source.

use std::path::Path;

use super::config_v2::GateConfig;
use super::method_result::If2Record;
use super::state::FitState;
use crate::chain_index::ChainIndex;
use crate::evidence::NATS_TO_DB;

/// Soft threshold: params between this and `gate.a_thresh` get a
/// prominent warning but refine still runs. When `gate.a_thresh ≤
/// A_SOFT` the SoftWarn band is empty (every above-soft Â also
/// exceeds the hard gate). Matches the existing scout diagnostic
/// colour-coding (red ≥ a_thresh, yellow A_SOFT..a_thresh,
/// green < A_SOFT).
pub const A_SOFT: f64 = 1.05;

/// Where one parameter's chain-agreement Â sits relative to **the gate that
/// will actually be applied to it**.
///
/// Derived from `GateConfig::a_thresh`, not from a literal, because every
/// renderer that spelled the band as a constant drifted from the gate the
/// moment `a_thresh` moved off that constant — and the default did move, to
/// 1.01. One `fit summary` printed `max Â = 1.030 ✗ (threshold 1.01)` in the
/// gate block and `Â=1.030 ✓` in the parameter table twenty lines below.
///
/// This is the **glyph** band and nothing else. The diagnostic severity ladder
/// — whether a finding is a warning or an error — is a different question with
/// a different answer (`DiagnosticKind::severity`), and merging the two would
/// make "the gate refuses this" and "this is an error-level finding" one knob
/// when they are two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementBand {
    /// Below [`A_SOFT`]: nothing to say.
    Pass,
    /// In `[A_SOFT, a_thresh)`: prominent warning, but refine still runs.
    /// Empty whenever `a_thresh <= A_SOFT`, which the 1.01 default is.
    SoftWarn,
    /// At or above `a_thresh` — the comparison
    /// [`check_scout_convergence`] refuses on.
    Fail,
    /// Â is not defined for this parameter: the within-chain variance `W`
    /// collapsed relative to the parameter's scale, so Â would diverge with no
    /// diagnostic meaning and `compute_chain_agreement` returns `NaN` (gh#45).
    ///
    /// **Not** a failure. Every comparison against `NaN` is false, so a band
    /// written as an if/else-if chain of `<` falls through to the failure arm
    /// and reports a parameter that was never assessed as one that failed. The
    /// compound gate's Δ_dB leg carries the verdict for these.
    NotAssessed,
}

impl AgreementBand {
    /// The plain glyph. Colour is the renderer's business — `fit summary`
    /// paints through its own `ok`/`warn`/`err`, the method report through raw
    /// ANSI — so this returns the character alone.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✓",
            Self::SoftWarn => "~",
            Self::Fail => "✗",
            Self::NotAssessed => "n/a",
        }
    }
}

impl GateConfig {
    /// Which band one parameter's Â falls in, under **this** gate.
    ///
    /// The single authority for the ✓/~/✗ on an Â, so a glyph cannot disagree
    /// with the verdict printed beside it. [`check_scout_convergence`] reduces
    /// through the same ladder.
    pub fn a_band(&self, a: f64) -> AgreementBand {
        if !a.is_finite() {
            AgreementBand::NotAssessed
        } else if a >= self.a_thresh {
            AgreementBand::Fail
        } else if a >= A_SOFT {
            AgreementBand::SoftWarn
        } else {
            AgreementBand::Pass
        }
    }
}

/// The noise floor of a PF-based log-likelihood estimator at reasonable
/// particle counts, in nats. A between-chain spread past three of these is
/// read as multi-modality in [`format_hard_verdict`].
pub const LOGLIK_EPSILON_MIN: f64 = 3.0;

/// Verdict from the compound convergence check. `SoftWarn` callers
/// should print the named parameters prominently. `Hard` and
/// `DecibansSpread` callers should error unless the user passed
/// `--allow-nonconverged-source`, in which case downgrade to a warning.
#[derive(Debug)]
pub enum ScoutGateVerdict {
    Ok,
    SoftWarn { param_agreement: Vec<(String, f64)> },
    Hard {
        /// Every param (initial-state ones included) with Â ≥ `gate.a_thresh`. Named and
        /// sorted worst-first so the error message leads with the
        /// most obvious failure.
        failing: Vec<(String, f64)>,
        /// Every dynamics (non-initial-state) Â, for the full diagnostic table.
        all_structural: Vec<(String, f64)>,
        /// Â for the `perturb_only_at_t0` (initial-state) params, gated like
        /// the rest and listed apart.
        perturb_only_at_t0: Vec<(String, f64)>,
        /// Spread across scout's per-chain final logliks. A wide
        /// spread is the strongest signal of multi-modality.
        loglik_spread: f64,
    },
    /// New in §Proposal 3 (Step 8): chain agreement Â passed but the
    /// inter-chain CLEAN-EVAL log-likelihood spread (in decibans)
    /// exceeded `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`.
    /// Strong signal that chains landed in different basins even with
    /// per-parameter trajectories that look stable.
    DecibansSpread {
        delta_db: f64,
        threshold_db: f64,
        sigma_max: f64,
        /// The clean-eval loglik of every chain that reached the eval, with
        /// the chain it belongs to when `fit_state.toml` records it.
        chain_logliks: SpreadChains,
    },
    /// No Â failed, but these parameters' Â is not finite (their
    /// within-chain variance collapsed), so the Â leg is incomplete. The
    /// decibans leg does not stand in for it: chains spread along a ridge can
    /// agree in log-likelihood while disagreeing in θ. Or the run lost
    /// started chains, so both legs judged the survivors only. Not a pass.
    Incomplete { params: Vec<String>, lost_chains: Option<LostChains> },
}

/// The per-chain clean-eval logliks behind a `DecibansSpread` refusal.
///
/// The spread itself is computable from the logliks alone, so a missing
/// `chain_eval_ids` must not let a wide spread pass. But a position in
/// `chain_eval_logliks` is not a chain: IF2 drops filter-degenerate chains
/// before the eval, so once any chain is missing, labelling by position names
/// the wrong `chain_N/` (gh#978). The two cases are therefore distinct values,
/// and only `Named` can name a chain.
#[derive(Debug, Clone, PartialEq)]
pub enum SpreadChains {
    /// `(chain, clean-eval loglik, its standard error)`, the chain carried
    /// from `FitState::chain_eval_ids` and the standard error of the
    /// replicate mean from `FitState::chain_eval_ses`.
    Named(Vec<(ChainIndex, f64, f64)>),
    /// The logliks only: `fit_state.toml` has no `chain_eval_ids`, or a count
    /// that disagrees with the logliks', so no loglik can be attributed.
    Unnamed(Vec<f64>),
}

/// The IF2 convergence assessment of one stored fit: both legs of the
/// compound gate, measured once. Every reader of an IF2 verdict — the
/// warm-start source check ([`check_scout_convergence`]), `fit table`
/// (`method_result::If2Run`) and `fit summary`'s gate block — reads
/// this, so the three cannot disagree about one fit.
///
/// The overall verdict judges every estimated parameter, initial-state
/// (`perturb_only_at_t0`) parameters included: an estimate whose chains
/// disagree is not converged, whatever IF2's perturbation schedule for it.
/// The two kinds are listed apart, so a renderer can also report the
/// dynamics-only sub-verdict ([`If2Assessment::dynamics_a_passes`]).
#[derive(Debug, Clone, PartialEq)]
pub struct If2Assessment {
    /// Finite Â of each dynamics (not `perturb_only_at_t0`) parameter, worst
    /// first.
    pub structural: Vec<(String, f64)>,
    /// Parameters (of either kind) whose Â is not finite — within-chain variance
    /// collapsed, so the statistic is unavailable there. Never a pass: the
    /// decibans leg does not stand in for it, because chains spread along a
    /// ridge can agree in log-likelihood while disagreeing in θ.
    pub unavailable: Vec<String>,
    /// Finite Â of each `perturb_only_at_t0` (initial-state) parameter,
    /// worst first.
    pub perturb_only_at_t0: Vec<(String, f64)>,
    /// The band of the worst finite Â over every parameter (`Pass` when there
    /// is none).
    pub a_band: AgreementBand,
    /// The decibans leg, when the leaf carries clean-eval logliks and SEs for
    /// at least two chains; `None` otherwise (the leg is not run).
    pub spread: Option<DecibansLeg>,
    /// Started chains the run lost (skipped mid-run when their particle
    /// filter degenerated), when there were any. Both legs above were then
    /// measured on the survivors only, so the verdict is incomplete, never a
    /// pass and never "not assessed": a fit that asked for several chains and
    /// finished with one has not shown that they agree.
    pub lost_chains: Option<LostChains>,
}

/// How many of the chains an IF2 run started reached the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LostChains {
    pub started: usize,
    pub finished: usize,
}

impl LostChains {
    /// From `fit_state.toml`'s `n_chains` and `n_good_chains`: `Some` only
    /// when fewer finished than started. A leaf that records no
    /// `n_good_chains` (written before IF2 recorded it) reads as none lost.
    pub fn from_state(state: &FitState) -> Option<Self> {
        Self::of(state.n_chains, state.n_good_chains)
    }

    /// From the started and finished chain counts.
    pub fn of(n_chains: usize, n_good_chains: Option<usize>) -> Option<Self> {
        let finished = n_good_chains?;
        (finished < n_chains).then_some(LostChains { started: n_chains, finished })
    }

    pub fn lost(&self) -> usize {
        self.started - self.finished
    }

    /// The sentence every refusal of such a source carries.
    pub fn describe(&self) -> String {
        let consequence = if self.finished == 1 {
            "one chain is left, and there is nothing to compare it with".to_string()
        } else {
            format!("chain agreement was judged on the {} survivors only", self.finished)
        };
        format!(
            "{} of its {} IF2 chains were lost (skipped mid-run when their particle filter \
             degenerated; `chain_starts.tsv` marks them `skipped`), so {consequence}",
            self.lost(), self.started)
    }
}

/// The inter-chain clean-eval spread against its SE-aware floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecibansLeg {
    pub delta_db: f64,
    /// `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`.
    pub threshold_db: f64,
    pub sigma_max: f64,
}

impl DecibansLeg {
    pub fn passes(&self) -> bool {
        self.delta_db < self.threshold_db
    }
}

impl If2Assessment {
    /// No finite Â, of any estimated parameter, is in the fail band.
    pub fn a_passes(&self) -> bool {
        self.a_band != AgreementBand::Fail
    }

    /// The dynamics-only sub-verdict: no finite Â of a non-initial-state
    /// parameter fails. Reported beside the overall verdict, never instead
    /// of it.
    pub fn dynamics_a_passes(&self, gate: &GateConfig) -> bool {
        self.structural.iter().all(|(_, a)| gate.a_band(*a) != AgreementBand::Fail)
    }

    /// Every finite Â, worst first: the parameters the verdict judges.
    pub fn all_finite(&self) -> Vec<(String, f64)> {
        let mut all: Vec<(String, f64)> =
            self.structural.iter().chain(&self.perturb_only_at_t0).cloned().collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all
    }

    /// Nothing was assessed: no Â of any kind (the run asked for one chain,
    /// or the leaf predates Â), no decibans leg, and no started chain lost.
    /// Not a pass; a starts-source check still accepts it, having no verdict
    /// to contradict.
    pub fn is_not_assessed(&self) -> bool {
        self.structural.is_empty()
            && self.perturb_only_at_t0.is_empty()
            && self.unavailable.is_empty()
            && self.spread.is_none()
            && self.lost_chains.is_none()
    }

    /// No finite Â failed, but some required Â is unavailable, or the run
    /// lost started chains: the verdict is incomplete, which is not a pass.
    pub fn is_incomplete(&self) -> bool {
        self.a_passes() && (!self.unavailable.is_empty() || self.lost_chains.is_some())
    }

    /// The compound verdict: every started chain finished, every required Â
    /// is available and passes, and the decibans leg passes where it was
    /// run. A leaf without clean-eval data is judged on Â alone.
    pub fn passes(&self) -> bool {
        !self.is_not_assessed()
            && self.lost_chains.is_none()
            && self.a_passes()
            && self.unavailable.is_empty()
            && self.spread.is_none_or(|leg| leg.passes())
    }

    /// The worst parameter, of either kind, and its Â.
    pub fn worst(&self) -> Option<(String, f64)> {
        self.all_finite().into_iter().next()
    }
}

/// Measure both legs of the IF2 gate on an IF2 leaf's record under `gate`.
/// The one measurement the warm-start source check, `fit summary` and
/// `fit table` read.
pub fn assess_if2(state: &If2Record, gate: &GateConfig) -> If2Assessment {
    let t0_set: std::collections::HashSet<&str> =
        state.perturb_only_at_t0_params.iter().map(|s| s.as_str()).collect();
    let mut structural: Vec<(String, f64)> = Vec::new();
    let mut unavailable: Vec<String> = Vec::new();
    let mut perturb_only_at_t0: Vec<(String, f64)> = Vec::new();
    for (name, &agreement) in &state.tail_chain_agreement {
        if !agreement.is_finite() {
            unavailable.push(name.clone());
        } else if t0_set.contains(name.as_str()) {
            perturb_only_at_t0.push((name.clone(), agreement));
        } else {
            structural.push((name.clone(), agreement));
        }
    }
    let worst_first = |a: &(String, f64), b: &(String, f64)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    structural.sort_by(worst_first);
    perturb_only_at_t0.sort_by(worst_first);

    let worst = structural.iter().chain(&perturb_only_at_t0)
        .map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let a_band = gate.a_band(worst);

    let spread = (state.chain_eval_logliks.len() >= 2
        && state.chain_eval_ses.len() == state.chain_eval_logliks.len())
    .then(|| {
        let hi = state.chain_eval_logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = state.chain_eval_logliks.iter().cloned().fold(f64::INFINITY, f64::min);
        let sigma_max = state.chain_eval_ses.iter().cloned().fold(0.0_f64, f64::max);
        DecibansLeg {
            delta_db: (hi - lo) * NATS_TO_DB,
            threshold_db: gate.decibans_thresh.max(8.0 * sigma_max * NATS_TO_DB),
            sigma_max,
        }
    });
    If2Assessment {
        structural, unavailable, perturb_only_at_t0, a_band, spread,
        lost_chains: LostChains::of(state.n_chains, state.n_good_chains),
    }
}

/// Check an IF2 leaf's convergence, for the warm-start source check.
///
/// Compound gate (Step 8, proposal §Proposal 3), measured by [`assess_if2`]:
///
/// 1. Chain-agreement Â on every estimated param, `perturb_only_at_t0`
///    ones included, must be `< gate.a_thresh`.
///    Failure → `Hard`. Â in `[A_SOFT, gate.a_thresh)` → `SoftWarn`.
/// 2. If both `chain_eval_logliks` and `chain_eval_ses` are populated
///    (≥ 2 entries), the inter-chain decibans-spread of clean-eval
///    log-likelihoods must be below
///    `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`. The
///    SE-aware floor prevents penalising chains whose Monte-Carlo
///    spread alone could explain the observed log-lik spread.
///    Failure → `DecibansSpread`, naming chains only when
///    `chain_eval_ids` matches the logliks ([`SpreadChains`]).
///
/// A run that lost started chains is `Incomplete` unless a leg failed
/// outright, Â or not: the survivors are all either leg saw.
///
/// No `tail_chain_agreement` and no lost chain — a run that asked for one
/// chain, or a leaf written before Â was recorded — returns `Ok`: there is
/// no verdict to contradict, and nothing is printed. Missing
/// `chain_eval_*` only skips the decibans leg.
pub fn check_scout_convergence(scout: &FitState, gate: &GateConfig) -> ScoutGateVerdict {
    let lost_chains = LostChains::from_state(scout);
    if scout.tail_chain_agreement.is_empty() {
        return match lost_chains {
            None => ScoutGateVerdict::Ok,
            Some(_) => ScoutGateVerdict::Incomplete { params: Vec::new(), lost_chains },
        };
    }
    let assessed = assess_if2(&If2Record::from_state(scout), gate);

    // Step 1 — Â check. Reduced through `GateConfig::a_band`, the same
    // authority every renderer uses, so a printed glyph and this verdict
    // cannot disagree about one number.
    match assessed.a_band {
        AgreementBand::Fail => {
            let failing: Vec<(String, f64)> = assessed.all_finite().into_iter()
                .filter(|(_, r)| gate.a_band(*r) == AgreementBand::Fail)
                .collect();
            let loglik_spread = if scout.chain_logliks.len() >= 2 {
                let hi = scout.chain_logliks.iter().cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let lo = scout.chain_logliks.iter().cloned()
                    .fold(f64::INFINITY, f64::min);
                hi - lo
            } else { 0.0 };
            return ScoutGateVerdict::Hard {
                failing,
                all_structural: assessed.structural,
                perturb_only_at_t0: assessed.perturb_only_at_t0,
                loglik_spread,
            };
        }
        // The soft band refuses nothing on its own, so the decibans leg and an
        // unavailable Â are checked first; the warning comes last, only for a
        // source that is otherwise accepted. The worst Â is a max over finite
        // entries seeded at 0.0, so `NotAssessed` is unreachable for it.
        AgreementBand::SoftWarn | AgreementBand::Pass | AgreementBand::NotAssessed => {}
    }

    // Step 2 — decibans-spread check on clean-eval logliks. Skipped
    // when the scout is pre-§Proposal 1 and didn't write the new fields.
    // The ids are not needed to measure the spread, so their absence does
    // not skip the check; it only stops the refusal naming chains (gh#978).
    if let Some(leg) = assessed.spread {
        if !leg.passes() {
            return ScoutGateVerdict::DecibansSpread {
                delta_db: leg.delta_db,
                threshold_db: leg.threshold_db,
                sigma_max: leg.sigma_max,
                // The leg only runs with one standard error per loglik.
                chain_logliks: if scout.chain_eval_ids.len() == scout.chain_eval_logliks.len() {
                    SpreadChains::Named(scout.chain_eval_ids.iter().copied()
                        .zip(scout.chain_eval_logliks.iter().copied())
                        .zip(scout.chain_eval_ses.iter().copied())
                        .map(|((c, ll), se)| (c, ll, se))
                        .collect())
                } else {
                    SpreadChains::Unnamed(scout.chain_eval_logliks.clone())
                },
            };
        }
    }

    if assessed.is_incomplete() {
        return ScoutGateVerdict::Incomplete {
            params: assessed.unavailable,
            lost_chains: assessed.lost_chains,
        };
    }

    // Only a source the assessment passes gets the soft-band warning; every
    // way of not passing has returned above.
    if assessed.a_band == AgreementBand::SoftWarn && assessed.passes() {
        let warnable: Vec<(String, f64)> = assessed.all_finite().into_iter()
            .filter(|(_, r)| gate.a_band(*r) == AgreementBand::SoftWarn)
            .collect();
        return ScoutGateVerdict::SoftWarn { param_agreement: warnable };
    }

    ScoutGateVerdict::Ok
}

/// Render the DecibansSpread verdict as a human error message.
/// Names the spread, the threshold (and which limb of `max(...)` it
/// came from), and the per-chain logliks in nats and decibans — each
/// against its own chain number — so the user can see whether one chain
/// is the obvious outlier and open the right `chain_N/` (gh#978). With
/// [`SpreadChains::Unnamed`] the logliks are listed without a chain and
/// the message says why.
pub fn format_decibans_spread_verdict(
    delta_db: f64,
    threshold_db: f64,
    sigma_max: f64,
    chain_logliks: &SpreadChains,
) -> String {
    let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
    let floor_source = if se_floor_db >= threshold_db {
        format!("8 · σ_max · NATS_TO_DB = 8 · {:.2} · {:.3} ≈ {:.1} dB",
            sigma_max, NATS_TO_DB, se_floor_db)
    } else {
        format!("user-configured floor decibans_thresh = {:.1} dB", threshold_db)
    };
    let mut msg = format!(
        "its chains landed in different basins.\n\n  \
         clean-eval log-likelihood spread:\n    \
         Δℓ = {:.2} dB > threshold = {:.2} dB ({})\n\n  \
         Per-chain clean logliks (nats / dB from worst):\n",
        delta_db, threshold_db, floor_source);
    let chain_logliks = match chain_logliks {
        SpreadChains::Named(v) => v.as_slice(),
        SpreadChains::Unnamed(v) => {
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for ll in &sorted {
                msg.push_str(&format!("    ℓ = {:>9.2}  ({:+.2} dB from worst)\n",
                    ll, (ll - lo) * NATS_TO_DB));
            }
            msg.push_str("\n  fit_state.toml records no `chain_eval_ids` matching these \
                logliks, so they cannot be attributed to chains; re-run that fit to \
                record which chain each belongs to.\n");
            msg.push_str(PICK_ONE);
            return msg;
        }
    };
    let lo = chain_logliks.iter().map(|(_, v, _)| *v).fold(f64::INFINITY, f64::min);
    let mut sorted: Vec<(ChainIndex, f64, f64)> = chain_logliks.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (chain, ll, se) in &sorted {
        msg.push_str(&format!("    chain {:<2}  ℓ = {:>9.2} ± {:.2}  ({:+.2} dB from worst)\n",
            chain, ll, se, (ll - lo) * NATS_TO_DB));
    }

    // Name the chains driving the spread, so the fix points at a chain index —
    // not just "the spread is wide". The same comparison the `fit summary`
    // per-chain table makes: each start's clean loglik against the pooled
    // mean of the others, over the evaluation noise of both (the standard
    // error of the clean-eval replicate mean). IF2 is an optimiser, so a
    // start that differs beyond that noise reached a different optimum. The
    // worst-chain fallback names the low end of a spread no single start
    // explains.
    use super::chain_diagnostics as cd;
    let estimates: Vec<cd::ChainLoglikEstimate> = chain_logliks.iter()
        .map(|&(chain, mean, se)| cd::ChainLoglikEstimate { chain, mean, weight: 1.0, se: Ok(se) })
        .collect();
    let scores = cd::assess_chain_logliks(&estimates);
    let flagged = cd::flagged_chains(&scores);
    let not_assessed: Vec<String> = scores.iter()
        .filter_map(|s| match &s.outcome {
            cd::ChainLoglikOutcome::NotAssessed(why) => Some(format!("chain {}, {}", s.chain, why.describe())),
            cd::ChainLoglikOutcome::Assessed(_) => None,
        })
        .collect();
    // An optimiser's starts are not symmetric: one above the others reached a
    // better optimum, one below a worse one. Both are named, apart.
    let side = |below: bool| -> Vec<String> {
        flagged.iter()
            .filter(|s| s.test().is_some_and(|t| (t.diff < 0.0) == below))
            .map(|s| format!("chain {}", s.chain))
            .collect()
    };
    if let Some(cutoff) = flagged.first().and_then(|s| s.test()).map(|t| t.cutoff) {
        msg.push('\n');
        for (below, word) in [(false, "above"), (true, "below")] {
            let chains = side(below);
            if !chains.is_empty() {
                msg.push_str(&format!(
                    "  Clean loglik {word} the other chains' beyond its evaluation noise \
                     (Z > {cutoff:.2}): {}\n",
                    chains.join(", ")));
            }
        }
    } else if let Some((worst, _, _)) = chain_logliks.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    {
        msg.push_str(&format!(
            "\n  Chain driving the spread (lowest clean loglik): chain {worst}\n"));
    }
    for line in &not_assessed {
        msg.push_str(&format!("  Not compared: {line}.\n"));
    }

    msg.push_str(PICK_ONE);
    msg
}

/// The remedies closing every `DecibansSpread` message.
const PICK_ONE: &str = "\n  Pick one:\n    \
                  - re-run it with more chains (the wider the spread,\n      \
                    the higher the chance one chain is in the right basin)\n    \
                  - inspect chain_evaluations.tsv to see which candidate\n      \
                    label dominated each chain — divergent labels are a\n      \
                    multimodality signal\n    \
                  - if the spread is genuinely Monte-Carlo noise, raise\n      \
                    [method.loglik_eval] n_particles or n_replicates\n    \
                  - relax the gate via [method.gate].decibans_thresh\n      \
                    or pass --decibans-thresh on the next run\n\n  \
                  To start from it anyway:  camdl fit run fit.toml --allow-nonconverged-source";

/// Render the Hard verdict as a human error message.
///
/// `gate` is the one this verdict came from, so every glyph in the table is
/// the same comparison the verdict is — not a literal that agrees only while
/// `a_thresh` happens to equal it.
pub fn format_hard_verdict(
    gate: &GateConfig,
    failing: &[(String, f64)],
    all_structural: &[(String, f64)],
    perturb_only_at_t0: &[(String, f64)],
    loglik_spread: f64,
    scout_best_loglik: f64,
    scout_best_chain_values: Option<&[(String, f64)]>,
) -> String {
    let mut msg = format!(
        "it did not converge.\n\n  \
         Tail Â (IF2 chain agreement over the last half of iterations), \
         threshold {:.2}:\n",
        gate.a_thresh);
    for (name, agreement) in all_structural {
        let band = gate.a_band(*agreement);
        let marker = if band == AgreementBand::Pass { " " } else { band.glyph() };
        msg.push_str(&format!("    {} {:<10} Â = {:>6.3}{}\n",
            marker, name, agreement,
            if band == AgreementBand::Fail {
                format!("   (>= {:.2})", gate.a_thresh)
            } else {
                String::new()
            }));
    }
    for (name, agreement) in perturb_only_at_t0 {
        let band = gate.a_band(*agreement);
        let marker = if band == AgreementBand::Pass { " " } else { band.glyph() };
        msg.push_str(&format!(
            "    {} {:<10} Â = {:>6.3}   (initial state, perturb_only_at_t0)\n",
            marker, name, agreement));
    }
    if loglik_spread > 0.0 {
        msg.push_str(&format!("\n  Loglik spread: {:.1} (best chain loglik {:.1})\n",
            loglik_spread, scout_best_loglik));
    }
    if loglik_spread > LOGLIK_EPSILON_MIN * 3.0 {
        msg.push_str("  -> likelihood surface is almost certainly multi-modal.\n");
    }
    msg.push_str(&format!("\n  Failing: {}\n",
        failing.iter().map(|(n, r)| format!("{} (Â={:.2})", n, r))
            .collect::<Vec<_>>().join(", ")));
    msg.push_str("\n  Pick one:\n    \
                  - re-run it with more chains or iterations\n    \
                  - narrow bounds to the basin its best chain found");
    if let Some(vals) = scout_best_chain_values {
        msg.push_str(":\n");
        for (name, value) in vals {
            msg.push_str(&format!("        {} ≈ {:.4}\n", name, value));
        }
        msg.push_str("      copy into [estimate.*] bounds / start values\n    ");
    } else {
        msg.push_str("\n    ");
    }
    msg.push_str("- fix a weakly identified initial-state parameter rather than \
                  estimate it\n\n  \
                  To start from it anyway (the starts may launder multi-modality):\n    \
                  camdl fit run fit.toml --allow-nonconverged-source");
    msg
}

/// Refuse a warm-start source whose stored verdict is not converged, unless
/// `allow` (`--allow-nonconverged-source`), in which case say so and proceed.
///
/// An nl-* leaf is judged by the verdict its optimizer recorded
/// (`FitState::optimizer_convergence`). An IF2 leaf carries
/// `tail_chain_agreement`, so it is judged by
/// the compound gate it was run under (`resolved_gate`, else the default). A
/// sampler leaf is judged by its stored R̂ through the same classification
/// `fit summary` renders: `max R̂` at or above the certification threshold,
/// or an unassessable R̂ (a sampler pathology), is not converged; a leaf whose
/// R̂ was never applicable (one chain) is not refused — there is no verdict to
/// contradict. `handle` names the source in the message.
pub fn check_source_converged(leaf: &Path, handle: &str, allow: bool) -> Result<(), String> {
    let dir = leaf.to_string_lossy().into_owned();
    use crate::run_meta::FitAlgorithm;
    // A verdict that cannot be read is refused, never read as converged; and
    // `allow` does not lift it — it accepts a verdict of "not converged", not
    // a missing one. Every fit method writes fit_state.toml, and `from_mle`
    // reads its estimate from it.
    let state = FitState::load(&dir).map_err(|e| format!(
        "starts source {handle}: cannot read its fit_state.toml, so its convergence \
         cannot be checked: {e}"))?;
    let verdict: Option<String> = match &state {
        // An nl-* leaf's `tail_chain_agreement` holds relative ranges, not
        // IF2's Â; it is judged by the optimizer's own recorded verdict.
        state if matches!(
            super::methods::parse_algorithm(&state.method),
            Some(FitAlgorithm::NlSbplx | FitAlgorithm::NlBobyqa | FitAlgorithm::NlLbfgs)
        ) => nlopt_source_verdict(state),
        // IF2, and a pre-split IF2 leaf (method `scout` / `refine`) that
        // carries Â. A one-chain IF2 leaf has no Â and no verdict to read.
        state if parse_algorithm_is(&state.method, FitAlgorithm::If2)
            || (super::methods::parse_algorithm(&state.method).is_none()
                && !state.tail_chain_agreement.is_empty()) => {
            let gate = state.resolved_gate.clone().unwrap_or_default();
            // A failed leg is refused for itself; when the run also lost
            // chains, the refusal says so, since the failure was measured on
            // the survivors only.
            let lost_note = |msg: String| match LostChains::from_state(state) {
                Some(lost) => format!("{msg}\n\n  Also: {}.", lost.describe()),
                None => msg,
            };
            match check_scout_convergence(state, &gate) {
                ScoutGateVerdict::Ok => None,
                ScoutGateVerdict::SoftWarn { param_agreement } => {
                    // Between the soft and hard bands: usable, but say which
                    // parameters the source's chains only nearly agreed on.
                    let named: Vec<String> = param_agreement
                        .iter()
                        .map(|(n, a)| format!("{n} (Â = {a:.3})"))
                        .collect();
                    crate::out::detached().say(format_args!(
                        "{w} starts source {handle}: chain agreement is \
                         below the certification band on {}",
                        named.join(", "), w = crate::out::paint::warning_tag()
                    ));
                    None
                }
                ScoutGateVerdict::Hard {
                    failing, all_structural, perturb_only_at_t0, loglik_spread,
                } => Some(lost_note(format_hard_verdict(
                    &gate, &failing, &all_structural, &perturb_only_at_t0,
                    loglik_spread, state.best_loglik, None,
                ))),
                ScoutGateVerdict::DecibansSpread {
                    delta_db, threshold_db, sigma_max, chain_logliks,
                } => Some(lost_note(format_decibans_spread_verdict(
                    delta_db, threshold_db, sigma_max, &chain_logliks,
                ))),
                ScoutGateVerdict::Incomplete { params, lost_chains } => {
                    let mut why: Vec<String> = Vec::new();
                    if let Some(lost) = lost_chains {
                        why.push(lost.describe());
                    }
                    if !params.is_empty() {
                        why.push(format!(
                            "chain agreement Â is unavailable (not finite; within-chain \
                             variance collapsed) for {}. The decibans leg does not stand in \
                             for it: chains spread along a ridge can agree in log-likelihood \
                             while disagreeing in θ",
                            params.join(", ")));
                    }
                    let mut msg = format!(
                        "its convergence verdict is incomplete: {}.", why.join("; and "));
                    if lost_chains.is_some() {
                        msg.push_str(
                            "\n\n  Re-run the source fit so that every chain finishes (more \
                             particles, or narrower bounds that keep the starts where the \
                             filter can follow the data).\n\n  \
                             To start from it anyway:  camdl fit run fit.toml \
                             --allow-nonconverged-source");
                    }
                    Some(msg)
                }
            }
        }
        state if matches!(
            super::methods::parse_algorithm(&state.method),
            Some(FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Mh)
        ) => sampler_source_verdict(leaf)
            .map_err(|e| format!("starts source {handle}: {e}"))?,
        // A pfilter leaf ran no search: there is nothing to converge.
        state if parse_algorithm_is(&state.method, FitAlgorithm::Pfilter) => None,
        state => return Err(format!(
            "starts source {handle}: its fit_state.toml names method `{}`, which this \
             camdl cannot judge, so its convergence cannot be checked",
            state.method)),
    };
    match verdict {
        None => Ok(()),
        Some(msg) if allow => {
            crate::out::detached().say(format_args!(
                "{w} starts source {handle}: {msg}\n  \
                 --allow-nonconverged-source: starting from it anyway.",
                w = crate::out::paint::warning_tag()
            ));
            Ok(())
        }
        Some(msg) => Err(format!("starts source {handle}: {msg}")),
    }
}

fn parse_algorithm_is(method: &str, want: crate::run_meta::FitAlgorithm) -> bool {
    super::methods::parse_algorithm(method) == Some(want)
}

/// An nl-* leaf's verdict: the optimizer's own, recorded at the end of its
/// run. A leaf written before the verdict was recorded cannot be judged, and
/// is refused rather than passed.
fn nlopt_source_verdict(state: &FitState) -> Option<String> {
    match &state.optimizer_convergence {
        Some(v) if v.passes() => None,
        Some(v) => {
            let mut legs = Vec::new();
            if !v.all_chains_converged() {
                legs.push(format!(
                    "{} of its chains ran out of evaluations and {} failed",
                    v.n_maxeval, v.n_failed));
            }
            if v.chain_agreement_failed {
                legs.push(format!(
                    "its chains ended apart (largest spread {:.1}% of a bound width, {:.3e} \
                     in natural units)", v.max_rel_range * 100.0, v.max_abs_range));
            }
            if v.decibans_failed {
                legs.push(format!(
                    "its converged chains' log-likelihoods differ by {:.1} nats",
                    v.delta_loglik));
            }
            Some(format!("its optimizer did not converge to one optimum: {}.",
                legs.join("; and ")))
        }
        None => Some(
            "it records no optimizer convergence verdict (written before camdl recorded \
             one), so its convergence cannot be checked. Re-run it with `--force`."
                .to_string(),
        ),
    }
}

/// A leaf's stored sampler verdict, through the same classification
/// `fit summary` renders. `Ok(None)` when converged, or when the leaf is not a
/// sampler or its R̂ was never applicable; `Err` when the record or the method
/// result cannot be read — an unreadable verdict is not a converged one.
fn sampler_source_verdict(leaf: &Path) -> Result<Option<String>, String> {
    use super::method_result::{MaxRhat, MethodResult, RHAT_CONVERGED_THRESHOLD};
    let rec = match runid::read_leaf(leaf) {
        Ok(Some(l)) => l.record,
        Ok(None) => return Err("it holds no run.json, so its convergence cannot be checked".into()),
        Err(u) => return Err(format!("{}, so its convergence cannot be checked", u.reason)),
    };
    let method = rec.inputs.get("method").and_then(|m| m.as_str())
        .ok_or("its run.json names no method, so its convergence cannot be checked")?
        .to_string();
    let typed = MethodResult::load_from(leaf, &method)
        .map_err(|e| format!("cannot load its {method} result, so its convergence cannot be checked: {e}"))?;
    let diag = match &typed {
        MethodResult::Pgas(r) => &r.diagnostics,
        MethodResult::Pmmh(r) => &r.diagnostics,
        _ => return Ok(None),
    };
    Ok(match diag.max_rhat_status() {
        MaxRhat::Reported(v) if v < RHAT_CONVERGED_THRESHOLD => None,
        MaxRhat::Reported(v) => Some(format!(
            "its posterior did not converge: max R̂ = {v:.3}, threshold {RHAT_CONVERGED_THRESHOLD}"
        )),
        MaxRhat::ChainsFrozen { chains } => Some(format!(
            "its posterior did not converge: {}, so its draws include a point mass",
            super::method_result::frozen_chains_clause(&chains)
        )),
        MaxRhat::Unassessable { params } => Some(format!(
            "its R̂ is unavailable for {} — a sampler failure, or a summary that does not \
             record it — so it is not certified",
            params.join(", ")
        )),
        MaxRhat::NotApplicable { .. } | MaxRhat::NoParams => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_leaf(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "camdl_gating_{tag}_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A source whose verdict cannot be read is refused, not read as
    /// converged — and `--allow-nonconverged-source` does not lift it: the
    /// flag accepts a verdict of "not converged", not a missing one.
    #[test]
    fn a_source_whose_verdict_cannot_be_read_is_refused() {
        // fit_state.toml present but corrupt.
        let corrupt = scratch_leaf("corrupt_state");
        std::fs::write(corrupt.join("fit_state.toml"), "method = \"pgas\"\nseed = [oops\n").unwrap();
        for allow in [false, true] {
            let err = check_source_converged(&corrupt, "@corrupt", allow)
                .expect_err("an unreadable fit_state.toml is refused");
            assert!(err.contains("@corrupt") && err.contains("fit_state.toml"), "{err}");
        }
        std::fs::remove_dir_all(&corrupt).ok();

        // A sampler leaf whose method result cannot be loaded (its
        // run.json names pgas, and its summary is missing).
        let sampler = scratch_leaf("no_summary");
        let mut st = make_state(&[], &[], &[-60.0, -61.0], -60.0);
        st.method = "pgas".into();
        st.save(&sampler.to_string_lossy()).unwrap();
        std::fs::write(sampler.join("run.json"), "{ not json").unwrap();
        let err = check_source_converged(&sampler, "@sampler", false)
            .expect_err("an unreadable run.json is refused");
        assert!(err.contains("@sampler"), "{err}");
        std::fs::remove_dir_all(&sampler).ok();
    }

    /// An nl-* leaf whose starts landed far apart. Its stored
    /// `tail_chain_agreement` holds per-parameter RELATIVE RANGES (range over
    /// bound width, at most 1 inside the box), not IF2's Â, so judging it
    /// against IF2's Â threshold (1.01) can never fail it. It must be judged
    /// by the optimizer's own two-leg verdict, which this leaf records as
    /// failed.
    #[test]
    fn an_nlopt_source_whose_starts_disagree_is_refused() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_gating_nl_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("fit_state.toml"), r#"
method = "nl-sbplx"
seed = 1
timestamp = "2026-10-29T00:00:00Z"
best_loglik = -50.0
initial_loglik = nan
best_chain = 0
n_chains = 3
n_good_chains = 3
chain_logliks = [-50.0, -95.0, -140.0]

[start_values]
beta = 0.4

[tail_chain_agreement]
beta = 0.6

[resolved_gate]
a_thresh = 1.01
decibans_thresh = 30.0

[optimizer_convergence]
n_converged = 3
n_maxeval = 0
n_failed = 0
max_rel_range = 0.6
max_abs_range = 2.9
chain_agreement_failed = true
delta_loglik = 90.0
decibans_failed = true
"#).unwrap();
        let got = check_source_converged(&dir, "@nl", false);
        std::fs::remove_dir_all(&dir).ok();
        let err = got.expect_err("an nl-* source whose chains disagree must be refused");
        assert!(err.contains("@nl"), "{err}");
    }

    fn make_state(
        tail_chain_agreement: &[(&str, f64)],
        perturb_only_at_t0_params: &[&str],
        chain_logliks: &[f64],
        best_loglik: f64,
    ) -> FitState {
        FitState {
            method: "scout".into(),
            seed: 1,
            timestamp: "2026-04-19T00:00:00Z".into(),
            input_hash: None, camdl_version: None,
            best_loglik,
            initial_loglik: f64::NEG_INFINITY,
            best_chain: ChainIndex(1),
            n_chains: chain_logliks.len().max(1),
            n_good_chains: None,
            start_values: Default::default(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: tail_chain_agreement.iter()
                .map(|(k, v)| (k.to_string(), *v)).collect(),
            perturb_only_at_t0_params: perturb_only_at_t0_params.iter()
                .map(|s| s.to_string()).collect(),
            chain_logliks: chain_logliks.to_vec(),
            chain_eval_logliks: vec![],
            chain_eval_ids: Vec::new(),
            chain_eval_ses: vec![],
            resolved_gate: None,
            resolved_loglik_eval: None,
            chain_init_source: None,
            chain_starts_kind: None,
            dt_check: None,
            pf_noise: None,
            optimizer_convergence: None,
        }
    }

    /// Legacy GateConfig matching the pre-§Proposal 3 thresholds —
    /// useful for tests that exercise the SoftWarn band and the
    /// initial-state exemption logic, both of which were defined before the new
    /// stricter default `a_thresh = 1.01`.
    fn legacy_gate() -> GateConfig {
        GateConfig { a_thresh: 1.10, decibans_thresh: f64::INFINITY }
    }

    #[test]
    fn hard_gate_fires_when_structural_agreement_exceeds_threshold() {
        let s = make_state(
            &[("beta", 3.5), ("gamma", 1.2), ("I0", 16.5)],
            &["I0"],
            &[-60.2, -62.5, -63.3, -64.5, -66.2, -68.7, -854.6],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Hard { failing, loglik_spread, .. } => {
                let names: Vec<&str> = failing.iter()
                    .map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"),
                    "beta (Â=3.5) must fail the gate: {:?}", names);
                assert!(names.contains(&"gamma"),
                    "gamma (Â=1.2) must fail: {:?}", names);
                // An initial-state (perturb_only_at_t0) param is gated too.
                assert!(names.contains(&"I0"),
                    "I0 (Â=16.5) is perturb_only_at_t0 and still fails: {:?}",
                    names);
                // Loglik spread computed: 854.6 − 60.2 = 794.4
                assert!((loglik_spread - 794.4).abs() < 0.1,
                    "loglik spread {:.1}, expected 794.4", loglik_spread);
            }
            other => panic!("expected Hard, got {:?}", other),
        }
    }

    #[test]
    fn perturb_only_at_t0_agreement_is_gated() {
        // All dynamics params are green; only the perturb_only_at_t0 params
        // have extreme Â. The check fails: an estimate whose chains disagree
        // is not converged, initial state or not.
        let s = make_state(
            &[("beta", 1.02), ("gamma", 1.01), ("I0", 16.5), ("R_init", 5.5)],
            &["I0", "R_init"],
            &[-60.2, -60.5],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Hard { failing, .. } => {
                let names: Vec<&str> = failing.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(names, vec!["I0", "R_init"], "worst first");
            }
            other => panic!("expected Hard on the initial-state params, got {:?}", other),
        }
        assert!(!assess_if2(&If2Record::from_state(&s), &legacy_gate()).a_passes());
        assert!(assess_if2(&If2Record::from_state(&s), &legacy_gate()).dynamics_a_passes(&legacy_gate()),
            "the dynamics-only sub-verdict passes");
    }

    /// A soft-band Â does not excuse the other legs: a failing decibans leg
    /// or an unavailable Â still refuses, as `fit table` reports `fail_db` /
    /// `incomplete` for the same leaf. The soft band used to return first.
    #[test]
    fn soft_warn_does_not_skip_the_decibans_leg_or_an_unavailable_a() {
        let mut s = make_state(&[("beta", 1.07), ("gamma", 1.02)], &[], &[-60.2, -160.5], -60.2);
        s.chain_eval_logliks = vec![-60.0, -160.0];
        s.chain_eval_ses = vec![0.5, 0.5];
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        assert!(matches!(check_scout_convergence(&s, &gate), ScoutGateVerdict::DecibansSpread { .. }),
            "{:?}", check_scout_convergence(&s, &gate));
        let mut s = make_state(&[("beta", 1.07), ("gamma", f64::NAN)], &[], &[-60.2, -60.5], -60.2);
        s.chain_eval_logliks.clear();
        assert!(matches!(check_scout_convergence(&s, &gate), ScoutGateVerdict::Incomplete { .. }),
            "{:?}", check_scout_convergence(&s, &gate));
    }

    #[test]
    fn soft_warn_between_thresholds() {
        let s = make_state(
            &[("beta", 1.07), ("gamma", 1.02)],
            &[],
            &[-60.2, -60.5],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::SoftWarn { param_agreement } => {
                let names: Vec<&str> = param_agreement.iter()
                    .map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"));
                assert!(!names.contains(&"gamma"),
                    "gamma (1.02) is below soft threshold, shouldn't be warned");
            }
            other => panic!("expected SoftWarn, got {:?}", other),
        }
    }

    #[test]
    fn legacy_state_with_no_agreement_returns_ok() {
        // Absent tail_chain_agreement (legacy fit_state from pre-2026-04-19):
        // caller treats this as "unknown, warn and proceed" via the
        // Ok verdict.
        let s = make_state(&[], &[], &[-60.0], -60.0);
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Ok => (),
            other => panic!("legacy state → Ok, got {:?}", other),
        }
    }

    /// Step 8: under the new strict default (`a_thresh = 1.01`),
    /// structural Â of 1.05 is already a hard fail — even though the
    /// pre-§Proposal 3 gate would have only soft-warned at this value.
    /// This is the intended tightening; documented here so a future
    /// reader doesn't try to "fix" it.
    #[test]
    fn default_gate_is_strict_about_chain_agreement() {
        let s = make_state(
            &[("beta", 1.05), ("gamma", 1.001)],
            &[],
            &[-60.0, -60.5],
            -60.0,
        );
        match check_scout_convergence(&s, &GateConfig::default()) {
            ScoutGateVerdict::Hard { failing, .. } => {
                let names: Vec<&str> = failing.iter().map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"),
                    "default a_thresh = 1.01 must fail beta (Â=1.05): {:?}", names);
            }
            other => panic!("expected Hard under default gate, got {:?}", other),
        }
    }

    /// Step 8 — SE-aware floor case from the handoff: small SE means
    /// `8·σ_max·NATS_TO_DB` is below the user-configured floor, so the
    /// floor binds. A spread of 100 dB exceeds 30 dB → DecibansSpread.
    #[test]
    fn decibans_spread_fails_when_floor_is_binding() {
        // Spread = 100 dB → 100 / NATS_TO_DB ≈ 23.03 nats.
        let mut s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -60.0],   // legacy chain_logliks unused by gate 2
            -60.0,
        );
        s.chain_eval_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_eval_ids = vec![ChainIndex(0), ChainIndex(1)];
        s.chain_eval_ses = vec![0.5, 0.5];   // 8·0.5·NATS_TO_DB ≈ 17.4 dB < 30 dB

        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::DecibansSpread {
                delta_db, threshold_db, sigma_max, chain_logliks,
            } => {
                assert!((delta_db - 100.0).abs() < 0.5,
                    "delta_db ≈ 100 dB; got {}", delta_db);
                assert!((threshold_db - 30.0).abs() < 1e-9,
                    "30 dB floor must bind (8·σ·NATS_TO_DB ≈ 17.4 dB < 30); got {}",
                    threshold_db);
                assert!((sigma_max - 0.5).abs() < 1e-12);
                assert!(matches!(&chain_logliks, SpreadChains::Named(v) if v.len() == 2),
                    "ids recorded, so the chains are named: {chain_logliks:?}");
            }
            other => panic!("expected DecibansSpread, got {:?}", other),
        }
    }

    /// Step 8 — SE-aware floor case from the handoff: large SE pushes
    /// the threshold above the user-configured floor (8·5·NATS_TO_DB
    /// ≈ 173.7 dB), so a 100 dB spread now passes.
    #[test]
    fn decibans_spread_passes_when_se_aware_floor_dominates() {
        let mut s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -60.0],
            -60.0,
        );
        s.chain_eval_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_eval_ids = vec![ChainIndex(0), ChainIndex(1)];
        s.chain_eval_ses = vec![5.0, 5.0];   // 8·5·NATS_TO_DB ≈ 173.7 dB

        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::Ok => (),
            other => panic!("expected Ok (SE-aware floor dominates), got {:?}", other),
        }
    }

    /// Step 8 — when clean-eval fields are absent (legacy fit_state
    /// or pre-§Proposal 1), the decibans check is skipped silently.
    /// The Â check still runs.
    #[test]
    fn missing_clean_eval_fields_skip_decibans_check() {
        let s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -200.0],   // huge legacy spread, but Step 2 doesn't use this
            -60.0,
        );
        match check_scout_convergence(&s, &GateConfig::default()) {
            ScoutGateVerdict::Ok => (),
            other => panic!("missing clean_eval fields → Ok; got {:?}", other),
        }
    }

    /// `(chain, loglik, se)` triples for contiguous chains counted from 0,
    /// every clean-eval loglik known to standard error 0.5.
    fn indexed(logliks: &[f64]) -> SpreadChains {
        indexed_se(logliks, 0.5)
    }

    fn indexed_se(logliks: &[f64], se: f64) -> SpreadChains {
        SpreadChains::Named(logliks.iter().enumerate().map(|(i, &v)| (ChainIndex(i), v, se)).collect())
    }

    /// The refusals a user reads name no stage of the retired staged
    /// pipeline: "scout" is not a thing a fit has any more, so "re-run the
    /// scout" sends the reader looking for one.
    #[test]
    fn the_refusals_name_no_retired_stage() {
        let named = format_decibans_spread_verdict(
            240.0, 30.0, 0.5, &indexed(&[-60.0, -61.0, -300.0]));
        let unnamed = format_decibans_spread_verdict(
            240.0, 30.0, 0.5, &SpreadChains::Unnamed(vec![-60.0, -300.0]));
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        let hard = format_hard_verdict(
            &gate, &[("beta".into(), 1.4)], &[("beta".into(), 1.4)], &[], 50.0, -60.0,
            Some(&[("beta".into(), 0.5)]));
        for msg in [named, unnamed, hard] {
            assert!(!msg.to_lowercase().contains("scout"), "{msg}");
        }
    }

    /// gh#406: the DecibansSpread message names the chain that differs, not
    /// just the aggregate spread. A start 240 nats below five that agree
    /// within their evaluation noise is named.
    #[test]
    fn decibans_message_names_a_clear_outlier_chain() {
        // Five chains near -60, chain 5 stuck at -300; six chains → z* = 3.76.
        let chain_logliks = indexed(&[-60.0, -60.4, -59.6, -60.5, -60.2, -300.0]);
        let msg = format_decibans_spread_verdict(240.0, 30.0, 0.5, &chain_logliks);
        assert!(
            msg.contains("Clean loglik below the other chains' beyond its evaluation noise \
                          (Z > 3.76): chain 5\n"),
            "message must name chain 5, and only it:\n{msg}"
        );
        assert!(!msg.contains("Clean loglik above"), "{msg}");
        assert!(msg.contains("chain 5   ℓ =   -300.00 ± 0.50"), "the noise is printed:\n{msg}");
    }

    /// gh#406: two co-stuck chains among six are both named, and the four
    /// that agree are not.
    #[test]
    fn decibans_message_names_both_co_stuck_chains() {
        // Four good (jittered) near -60, two stuck near -400 (chains 4 and 5).
        let chain_logliks = indexed(&[-60.0, -60.4, -59.6, -60.5, -400.0, -401.0]);
        let msg = format_decibans_spread_verdict(340.0, 30.0, 0.5, &chain_logliks);
        let outliers = msg.lines().find(|l| l.contains("beyond its evaluation noise"))
            .unwrap_or_else(|| panic!("both co-stuck chains must be flagged:\n{msg}"));
        assert!(!outliers.contains("chain 0") && !outliers.contains("chain 3"), "{msg}");
        assert!(outliers.contains("chain 4"), "must name chain 4:\n{msg}");
        assert!(outliers.contains("chain 5"), "must name chain 5:\n{msg}");
        assert!(
            !msg.contains("Chain driving the spread"),
            "the fallback must NOT fire when robust outliers are found:\n{msg}"
        );
    }

    /// gh#406: the fallback. A spread that no single start explains beyond
    /// its evaluation noise (here each clean loglik is uncertain to 15 nats)
    /// still triggers the gate, so the message names the single worst chain
    /// driving the low end — always actionable.
    #[test]
    fn decibans_message_fallback_names_worst_on_smooth_spread() {
        let chain_logliks = indexed_se(&[-60.0, -70.0, -80.0, -90.0, -100.0, -110.0], 15.0);
        let SpreadChains::Named(named) = &chain_logliks else { unreachable!() };
        use super::super::chain_diagnostics as cd;
        let scores = cd::assess_chain_logliks(&named.iter()
            .map(|&(chain, mean, se)| cd::ChainLoglikEstimate { chain, mean, weight: 1.0, se: Ok(se) })
            .collect::<Vec<_>>());
        assert!(
            scores.iter().all(|s| s.test().is_some_and(|t| !t.flagged)),
            "precondition: no start differs beyond its noise: {scores:?}"
        );
        let msg = format_decibans_spread_verdict(150.0, 30.0, 15.0, &chain_logliks);
        assert!(
            msg.contains("Chain driving the spread (lowest clean loglik): chain 5"),
            "fallback must name the worst chain (-110 → chain 5):\n{msg}"
        );
        assert!(!msg.contains("beyond its evaluation noise"), "no flagged-chain line on a smooth spread:\n{msg}");
    }

    /// gh#978: the refusal names chains by their id, not by their position in
    /// `chain_eval_logliks`. IF2 drops PF-degenerate chains before the
    /// loglik-eval, so once any chain is missing a position is not a chain,
    /// and the user who opens the named `chain_N/` reads the wrong one.
    #[test]
    fn decibans_refusal_names_the_low_chain_by_its_id_not_its_position() {
        let mut s = make_state(&[("beta", 1.001)], &[], &[-60.0, -60.0], -60.0);
        // Chain 1 died in the filter; the chains that reached the eval are 0 and 2.
        s.chain_eval_ids = vec![ChainIndex(0), ChainIndex(2)];
        s.chain_eval_logliks = vec![-60.0, -300.0];
        s.chain_eval_ses = vec![0.5, 0.5];
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        let msg = match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::DecibansSpread {
                delta_db, threshold_db, sigma_max, chain_logliks,
            } => format_decibans_spread_verdict(delta_db, threshold_db, sigma_max, &chain_logliks),
            other => panic!("expected DecibansSpread, got {other:?}"),
        };
        assert!(
            msg.contains("Clean loglik below the other chains' beyond its evaluation noise \
                          (Z > 3.48): chain 2\n"),
            "the low chain is chain 2 (its directory is chain_2/), not row 1:\n{msg}"
        );
        assert!(msg.contains("Clean loglik above the other chains' beyond its evaluation noise \
                              (Z > 3.48): chain 0\n"), "{msg}");
        assert!(
            !msg.contains("chain 1"),
            "chain 1 never reached the eval and must not be named:\n{msg}"
        );
    }

    /// Without the ids the spread is still measurable, so a wide one is still
    /// refused — a missing field must not certify convergence. The refusal
    /// lists the logliks without naming a chain, since a position in
    /// `chain_eval_logliks` is not a chain (gh#978).
    #[test]
    fn decibans_leg_refuses_without_the_chain_ids_and_names_no_chain() {
        let mut s = make_state(&[("beta", 1.001)], &[], &[-60.0, -60.0], -60.0);
        s.chain_eval_logliks = vec![-60.0, -300.0];
        s.chain_eval_ses = vec![0.5, 0.5];
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        let msg = match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::DecibansSpread {
                delta_db, threshold_db, sigma_max, chain_logliks,
            } => {
                assert_eq!(chain_logliks, SpreadChains::Unnamed(vec![-60.0, -300.0]));
                format_decibans_spread_verdict(delta_db, threshold_db, sigma_max, &chain_logliks)
            }
            other => panic!("expected DecibansSpread without ids, got {other:?}"),
        };
        assert!(!msg.contains("chain 0") && !msg.contains("chain 1"),
            "no chain may be named without the ids:\n{msg}");
        assert!(msg.contains("cannot be attributed to chains"), "{msg}");
        assert!(msg.contains("-300.00"), "the logliks are still listed:\n{msg}");
    }
}
