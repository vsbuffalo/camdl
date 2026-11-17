//! The convergence types of a posterior fit: each parameter's R̂ and
//! effective sample sizes, the estimator's refusals, the bands and verdicts
//! read off them, and the chain number they name chains by.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Why a rank-normalized diagnostic could not be computed for a parameter.
///
/// Every variant names a property of the *input*, **with the numbers**. Callers
/// render these; they must never be collapsed to a bare `NaN`, which reads as a
/// numerical failure and hides which precondition was missed.
///
/// Serialized so the numbers survive the trip to `*_summary.json` and
/// `camdl fit summary` can print the same sentence the run printed at the
/// time. [`RhatRefusal`] is the lossy classification of the same fact — it
/// answers "is this a sampler pathology"; this type answers "what exactly was
/// wrong with the input".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "refusal", rename_all = "snake_case")]
pub enum ConvergenceError {
    /// R̂ compares chains; one chain has nothing to compare against.
    TooFewChains { n_chains: usize },
    /// Fewer than four draws per chain — below the structural minimum for a
    /// split-half statistic.
    TooFewDraws { n_draws: usize },
    /// The between-chain variance formula uses one draw count for every chain.
    UnequalChainLengths { expected: usize, chain: usize, found: usize },
    /// A draw is `NaN` or `±inf`. Rank normalization has no ordering for these
    /// and would silently propagate `NaN` through every statistic — see gh#607,
    /// a chain that recorded `log_posterior = −inf` for thousands of sweeps.
    ///
    /// `value` is non-finite by construction — that is what the variant is for
    /// — and `serde_json` writes any non-finite `f64` as `null`, which will not
    /// read back. It is carried as `"inf"` / `"-inf"` / `"NaN"` by
    /// [`crate::num::json_f64`], so the round-trip is exact.
    NonFiniteDraw {
        chain: usize,
        draw: usize,
        #[serde(with = "crate::num::json_f64")]
        value: f64,
    },
    /// Every draw of every chain is the same value to within
    /// `DEGENERATE_REL_TOL` of the parameter's own scale. The total variance
    /// is zero, so R̂'s denominator is zero and the rank transform is constant.
    ConstantDraws { value: f64 },
    /// Every chain began at one point (`starts = "single"`, `from_mle`,
    /// `from_params`). The estimator runs, but chains that began together
    /// agree by construction, so the between-chain statistic cannot say
    /// whether the posterior was explored; the reader that knows how the
    /// chains were started raises this instead of reporting the number.
    /// `starts` is the rule as written, for the report.
    PointStart { n_chains: usize, starts: String },
}

/// Why a parameter has no R̂, and — the part that decides what a fit may
/// claim — **whether that indicates a problem**.
///
/// The two are not the same question and collapsing them is how a fit that
/// could not be assessed came to report `converged: true`. A run given two
/// chains of three draws was never *offered* the shape a between-chain
/// statistic needs; a run whose sampler never accepted a move was, and failed.
/// The first is "not assessed", the second is "did not converge", and neither
/// is "converged".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RhatRefusal {
    /// A draw was `NaN` or `±inf` — gh#607's chain recording
    /// `log_posterior = −inf` for thousands of sweeps.
    NonFiniteDraw,
    /// R̂ itself evaluated non-finite. Every chain is internally constant at a
    /// value the others do not share, so the within-chain variance is zero and
    /// the between-chain variance is not: the 0%-acceptance deadlock.
    NonFiniteRhat,
    /// An **estimated** parameter that never moved. Distinct from a parameter
    /// the model pins, which is not offered to the estimator at all.
    ConstantDraws,
    /// Fewer than two chains.
    TooFewChains,
    /// Fewer than four draws per chain.
    TooFewDraws,
    /// Chains of differing length.
    UnequalChainLengths,
    /// The estimated parameter set could not be determined, so a constant
    /// column cannot be told from a pinned one and no honest classification is
    /// available.
    EstimatedSetUnknown,
    /// Every chain began at one point, so the statistic was never given the
    /// spread it needs. Not a pathology: writing the point rule was the
    /// acknowledgement (proposal 2026-09-08, §3.4).
    PointStart,
}

impl RhatRefusal {
    /// `true` when the refusal is evidence the sampler MISBEHAVED, so a fit
    /// carrying it must not be called converged. `false` when the run was
    /// simply never given the shape the statistic needs — then the honest
    /// report is "not assessed", which is also not "converged".
    pub fn is_pathology(self) -> bool {
        match self {
            Self::NonFiniteDraw | Self::NonFiniteRhat | Self::ConstantDraws => true,
            Self::TooFewChains
            | Self::TooFewDraws
            | Self::UnequalChainLengths
            | Self::EstimatedSetUnknown
            | Self::PointStart => false,
        }
    }

    /// One clause naming what happened, for a report.
    pub fn describe(self) -> &'static str {
        match self {
            Self::NonFiniteDraw => "a draw was NaN or infinite",
            Self::NonFiniteRhat => "every chain sat at its own single value",
            Self::ConstantDraws => "the parameter never moved",
            Self::TooFewChains => "fewer than 2 chains",
            Self::TooFewDraws => "fewer than 4 draws per chain",
            Self::UnequalChainLengths => "chains of differing length",
            Self::EstimatedSetUnknown => "the estimated parameter set is unknown",
            Self::PointStart => "every chain started at one point",
        }
    }
}

impl ConvergenceError {
    /// How this refusal should be classified for reporting.
    pub fn refusal(&self) -> RhatRefusal {
        match self {
            Self::TooFewChains { .. } => RhatRefusal::TooFewChains,
            Self::TooFewDraws { .. } => RhatRefusal::TooFewDraws,
            Self::UnequalChainLengths { .. } => RhatRefusal::UnequalChainLengths,
            Self::NonFiniteDraw { .. } => RhatRefusal::NonFiniteDraw,
            Self::ConstantDraws { .. } => RhatRefusal::ConstantDraws,
            Self::PointStart { .. } => RhatRefusal::PointStart,
        }
    }
}

impl fmt::Display for ConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewChains { n_chains } => write!(
                f, "R̂ needs at least 2 chains; got {n_chains}"),
            Self::TooFewDraws { n_draws } => write!(
                f, "split-R̂ needs at least 4 draws per chain; got {n_draws}"),
            Self::UnequalChainLengths { expected, chain, found } => write!(
                f, "chain {chain} has {found} draws, expected {expected} \
                    (the between-chain variance uses one draw count)"),
            Self::NonFiniteDraw { chain, draw, value } => write!(
                f, "chain {chain} draw {draw} is {value}; rank normalization \
                    is undefined for non-finite draws"),
            Self::ConstantDraws { value } => write!(
                f, "every draw is {value}: the parameter did not move, so R̂ \
                    has no within-chain variance to divide by"),
            Self::PointStart { n_chains, starts } => write!(
                f, "all {n_chains} chains started at one point (starts = {starts})"),

        }
    }
}

impl std::error::Error for ConvergenceError {}

/// Which half of `max(rhat_bulk, rhat_folded)` set the headline R̂ — the
/// answer to *why* R̂ is high, which the headline alone cannot give.
///
/// The two halves differ by exactly one transformation (folding about the
/// median), so the larger one names which kind of between-chain disagreement
/// the statistic is reacting to. That is a decomposition, not a threshold: no
/// cutoff on the gap is proposed or applied here, deliberately — see
/// `docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md`, which
/// defers the lint until the gaps have been observed on a corpus of real fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhatDriver {
    /// `rhat_bulk` is the larger: the chains disagree about **where** the
    /// posterior sits. A chain drifting across its own run reads this way,
    /// because splitting compares each chain's halves.
    Location,
    /// `rhat_folded` is the larger: the chains agree on location and disagree
    /// on **spread**, which `rhat_bulk` cannot see.
    Scale,
}

impl RhatDriver {
    /// The larger half, or `None` when either is undefined (so there is no
    /// comparison to report) — the folded half is undefined whenever
    /// `|x − median(x)|` is constant.
    pub fn of(rhat_bulk: f64, rhat_folded: f64) -> Option<Self> {
        if rhat_bulk.is_nan() || rhat_folded.is_nan() {
            return None;
        }
        Some(if rhat_folded > rhat_bulk { Self::Scale } else { Self::Location })
    }

    /// One clause naming what the larger half means, for a report.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Location => "the chains disagree about where the posterior sits",
            Self::Scale => "the chains agree on location and disagree on spread",
        }
    }

    /// The name of the half, for a line that also prints both numbers.
    pub fn half(self) -> &'static str {
        match self {
            Self::Location => "bulk",
            Self::Scale => "folded",
        }
    }
}

/// Compound convergence gate verdict (the IF2 "is this method
/// converged" answer). String projection used in
/// `table_row.gate_verdict`:
///
/// | variant   | string       |
/// |-----------|--------------|
/// | `Pass`    | `"pass"`     |
/// | `FailA`   | `"fail_a"`   |
/// | `FailDb`  | `"fail_db"`  |
/// | `FailBoth`| `"fail_both"`|
///
/// Bayesian rows render `"n/a"` because the IF2 gate doesn't apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Pass,
    FailA,
    FailDb,
    FailBoth,
    /// No Â failed, but some required Â is unavailable (not finite;
    /// within-chain variance collapsed). Not a pass: the decibans leg does not
    /// stand in for a missing Â.
    Incomplete,
    /// Nothing was assessed: one chain, so there is no between-chain
    /// agreement or spread to judge. Not a pass.
    NotAssessed,
}

// String projection for the four variants is supplied by the
// `serde(rename_all = "snake_case")` derive: serializing a
// `GateVerdict` produces exactly `"pass"` / `"fail_a"` / `"fail_db"`
// / `"fail_both"`. A separate `as_str` method would be redundant
// with that and prone to drift; consumers call `serde_json::to_value`
// (or destructure on the variant) instead.

/// Posterior convergence + efficiency diagnostics, shared by every Bayesian
/// sampler (PGAS, PMMH, mh). Every sampler computes these the same way
/// — per-param rank-normalized R̂ and bulk/tail ESS via
/// `fit::runner::compute_rhat_ess` in `cli`
/// — and every renderer reads the same accessors below. Adding a new Bayesian
/// method means *filling this struct*, not re-deriving efficiency metrics per
/// method (the divergence that let a per-method allowlist drop `mh`).
///
/// One convergence statistic, in the three states it can actually be in.
///
/// Not `Option<f64>`, and not a bare `f64`. `serde_json` writes any non-finite
/// `f64` as `null`, so `Some(f64::INFINITY)` and `None` are indistinguishable
/// once a summary round-trips through disk — which would silently collapse
/// "∞, the sampler never moved" into "not computed". The three states have to
/// be named to survive serialization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stat {
    /// A finite measurement.
    Value(f64),
    /// Mathematically `+∞`: R̂ with zero within-chain variance, i.e. every
    /// chain sat at its own single value. A real answer, not a failure.
    Infinite,
    /// Not defined for this input — R's `NA`. The folded half of R̂ is
    /// undefined whenever `|x − median(x)|` is constant, and tail-ESS is
    /// undefined when a tail indicator is constant.
    Undefined,
}

impl Stat {
    /// From a raw `f64`, mapping the two non-finite cases onto their names.
    pub fn from_f64(v: f64) -> Self {
        if v.is_finite() {
            Stat::Value(v)
        } else if v.is_infinite() {
            Stat::Infinite
        } else {
            Stat::Undefined
        }
    }

    /// The finite value, if there is one. `Infinite` yields `None` — callers
    /// that must distinguish "no number" from "unboundedly bad" match instead.
    pub fn finite(self) -> Option<f64> {
        match self {
            Stat::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Rendered for a table cell: the number, `∞`, or `missing`.
    pub fn cell(self, precision: usize, missing: &str) -> String {
        match self {
            Stat::Value(v) => format!("{:.*}", precision, v),
            Stat::Infinite => "∞".to_string(),
            Stat::Undefined => missing.to_string(),
        }
    }
}

/// What the estimator produced for one parameter.
///
/// A sum type because the two cases have disjoint payloads: there is no R̂ to
/// hold when the reason it is missing is the thing being reported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ParamConvergence {
    /// The estimator ran. Individual statistics may still be `Infinite` or
    /// `Undefined`; each says so for itself.
    Scored {
        /// `max(rank-normalized split-R̂, folded split-R̂)` — the headline.
        rhat: Stat,
        /// The **location** half of that max: rank-normalized split-R̂ without
        /// the fold. `Undefined` on a fit written before this was stored.
        #[serde(default = "undefined_stat")]
        rhat_bulk: Stat,
        /// The **spread** half: the same statistic on `|x − median(x)|`.
        /// Which of the two is larger is the answer to *why* R̂ is high, and
        /// the two remedies are different — see
        /// `docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md`.
        #[serde(default = "undefined_stat")]
        rhat_folded: Stat,
        /// Gelman & Rubin (1992), unsplit and on the raw scale. Kept because
        /// the rank-normalized statistic is BOUNDED (ceiling ~1.85 for two
        /// chains, ~4.5 for eight) and so cannot express severity, while this
        /// one can. See the 2026-08-22 proposal.
        rhat_classic: Stat,
        /// Rank-normalized bulk ESS.
        ess_bulk: Stat,
        /// Tail ESS: the smaller of the 5% and 95% indicator ESS.
        ess_tail: Stat,
        /// Per-chain Geyer ESS, one entry per chain, in chain order.
        ///
        /// Answers the follow-up question to a high R̂, which no cross-chain
        /// statistic can: are the chains each mixing well *within their own
        /// mode* (large and similar per-chain values — a multimodality
        /// problem) or is one of them stuck (one small value — a starting
        /// point or step-size problem)? Empty on a fit written before this was
        /// stored.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        ess_per_chain: Vec<f64>,
        /// Every chain sat at its own single value — the sampler never
        /// accepted a move. R̂ is then `∞` or `Undefined`; this is what lets a
        /// report say *why* rather than printing an infinity and leaving the
        /// reader to infer the cause.
        #[serde(default, skip_serializing_if = "core::ops::Not::not")]
        all_chains_frozen: bool,
        /// The R̂ was withheld for a reason that is a property of how the
        /// chains were started — every chain at one point (proposal
        /// 2026-09-08, §3.4) — so the four R̂ fields are `Undefined` while the
        /// ESS fields, which the start does not invalidate the same way, are
        /// kept. Carries the refusal with its numbers for the report.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rhat_withheld: Option<ConvergenceError>,
    },
    /// The estimator could not run at all — a structural precondition, or a
    /// non-finite draw.
    NotScored {
        /// The classification: is this a sampler pathology or a shape the run
        /// was never given? [`MaxRhat`] keys on it.
        reason: RhatRefusal,
        /// The same refusal **with its numbers** — "R̂ needs at least 2 chains;
        /// got 1" rather than "fewer than 2 chains". `None` when the refusal
        /// was derived from a non-finite R̂ rather than raised by
        /// `rank_convergence`, and on a fit written before this was stored.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ConvergenceError>,
    },
}

/// `serde` default for the two R̂ halves: a fit written before they were
/// stored has no value, which is `Undefined`, not zero.
fn undefined_stat() -> Stat {
    Stat::Undefined
}

impl ParamConvergence {
    /// The headline R̂, or `Undefined` when the parameter was never scored.
    pub fn rhat(&self) -> Stat {
        match self {
            Self::Scored { rhat, .. } => *rhat,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// Bulk ESS, or `Undefined` when the parameter was never scored.
    pub fn ess_bulk(&self) -> Stat {
        match self {
            Self::Scored { ess_bulk, .. } => *ess_bulk,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// Tail ESS, or `Undefined` when the parameter was never scored.
    pub fn ess_tail(&self) -> Stat {
        match self {
            Self::Scored { ess_tail, .. } => *ess_tail,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// The **location** half of `max(rhat_bulk, rhat_folded)`: rank-normalized
    /// split-R̂ without the fold. `Undefined` when the parameter was never
    /// scored, or on a fit written before the halves were stored.
    pub fn rhat_bulk(&self) -> Stat {
        match self {
            Self::Scored { rhat_bulk, .. } => *rhat_bulk,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// The **spread** half: the same statistic on `|x − median(x)|`.
    /// `Undefined` on the same two conditions as [`rhat_bulk`](Self::rhat_bulk),
    /// and additionally whenever `|x − median(x)|` is constant.
    pub fn rhat_folded(&self) -> Stat {
        match self {
            Self::Scored { rhat_folded, .. } => *rhat_folded,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// Gelman & Rubin (1992), unsplit and on the raw scale. Kept beside the
    /// headline because the rank-normalized statistic is bounded and so cannot
    /// express severity, while this one can — a large gap between them says the
    /// disagreement lives in the tails.
    pub fn rhat_classic(&self) -> Stat {
        match self {
            Self::Scored { rhat_classic, .. } => *rhat_classic,
            Self::NotScored { .. } => Stat::Undefined,
        }
    }

    /// Which of the two halves set the headline, as a value rather than a
    /// sentence — the column a table shows beside both numbers. `None` when
    /// either half is undefined, so there is no comparison to report.
    pub fn rhat_driver(&self) -> Option<RhatDriver> {
        let Self::Scored { rhat_bulk, rhat_folded, .. } = self else {
            return None;
        };
        RhatDriver::of(rhat_bulk.finite()?, rhat_folded.finite()?)
    }

    /// Per-chain Geyer ESS, one entry per chain, in chain order.
    ///
    /// Empty for an unscored parameter and on a fit written before it was
    /// stored. The spread is the point — similar large values mean each chain
    /// mixes well inside its own mode, one small value means that chain is
    /// stuck, and those call for different fixes.
    pub fn ess_per_chain(&self) -> &[f64] {
        match self {
            Self::Scored { ess_per_chain, .. } => ess_per_chain,
            Self::NotScored { .. } => &[],
        }
    }

    /// One clause saying why this parameter carries no usable R̂, or `None`
    /// when it carries one.
    ///
    /// When the refusal's numbers survived to disk they are what is said —
    /// "chain 1 draw 7 is -inf; rank normalization is undefined for non-finite
    /// draws" locates the problem, where "a draw was NaN or infinite" only
    /// classifies it.
    pub fn why_no_rhat(&self) -> Option<String> {
        match self {
            Self::Scored { rhat_withheld: Some(e), .. } => Some(e.to_string()),
            Self::NotScored { detail: Some(e), .. } => Some(e.to_string()),
            Self::NotScored { reason, detail: None } => Some(reason.describe().to_string()),
            Self::Scored { all_chains_frozen: true, .. } => Some(
                "every chain sat at its own single value — the sampler never \
                 accepted a move".to_string(),
            ),
            Self::Scored { rhat: Stat::Undefined, .. } => Some(
                "the folded half of the statistic is undefined for this \
                 marginal".to_string(),
            ),
            Self::Scored { .. } => None,
        }
    }

    /// Whether this parameter is evidence the sampler MISBEHAVED, as opposed to
    /// simply not having been given the shape a between-chain statistic needs.
    pub fn is_pathology(&self) -> bool {
        match self {
            Self::NotScored { reason, .. } => reason.is_pathology(),
            Self::Scored { all_chains_frozen, rhat, .. } => {
                *all_chains_frozen || matches!(rhat, Stat::Infinite)
            }
        }
    }
}

/// The R̂ below which a Bayesian fit is reported as converged — the
/// machine-readable `converged` column and `fit summary`'s ✓/✗.
///
/// Named once so the gh#84 threshold decision (Vehtari et al. recommend 1.01
/// for the rank-normalized statistic, plus ESS > 400 before R̂ is read at all)
/// is a one-line change rather than a hunt through four call sites. Its VALUE
/// is unchanged and deliberately not part of that review's scope.
pub const RHAT_CONVERGED_THRESHOLD: f64 = 1.05;

/// The R̂ above which a parameter draws a `RhatHigh` diagnostic and the
/// end-of-run report marks it.
///
/// Unchanged in value from what every Bayesian method has always applied; what
/// changed in gh#84 is the STATISTIC — `RhatEss::rhat` (in `cli`'s
/// `fit::runner`) is now the rank-normalized split statistic of Vehtari et al.
/// (2021) rather than the classic Gelman & Rubin (1992) one, so the same 1.1
/// is a stricter bar.
/// Vehtari et al. recommend 1.01 for the rank-normalized statistic; adopting
/// that is a policy decision about what camdl certifies, tracked on gh#84, and
/// is deliberately NOT taken here.
pub const RHAT_REPORT_THRESHOLD: f64 = 1.1;

/// The R̂ above which a surface marks a parameter ✗ rather than ~ — "not
/// converged" against "badly enough that the chains are somewhere else".
/// Display only: nothing keys on it, and it draws no diagnostic of its own.
pub const RHAT_SEVERE_THRESHOLD: f64 = 1.5;

/// Where a reported R̂ sits against the bar camdl **certifies** against.
///
/// One band for every surface that glyphs an R̂. Two surfaces used to spell it
/// differently — the end-of-run block against `RHAT_REPORT_THRESHOLD` (1.1,
/// which is the bar that draws a *finding*), `fit summary` against
/// `RHAT_CONVERGED_THRESHOLD` (1.05, the bar `converged_at` and the
/// machine-readable `converged` column key on). A parameter in `[1.05, 1.1)`
/// therefore printed ✓ when the fit finished and ✗ when the user ran
/// `fit summary` on the same directory.
///
/// Certification and severity are separate questions and stay separate: this
/// band answers "may this be reported as converged", and
/// `DiagnosticKind::severity` answers "is the finding a warning or an error".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RhatBand {
    /// Below [`RHAT_CONVERGED_THRESHOLD`]: may be reported as converged.
    Converged,
    /// Between the two thresholds: not converged.
    NotConverged,
    /// At or above [`RHAT_SEVERE_THRESHOLD`]: the chains are somewhere else.
    Severe,
    /// R̂ is not a finite number, so there is no band. NOT a pass — the caller
    /// renders the reason ([`ParamConvergence::why_no_rhat`]) instead of a
    /// glyph.
    NotAssessed,
}

impl RhatBand {
    /// The band an R̂ falls in.
    ///
    /// The two non-finite cases are NOT the same and must not share an arm.
    /// `+∞` is a real answer and the worst one — every chain sat at its own
    /// single value, the 0%-acceptance deadlock — so it belongs in `Severe`;
    /// collapsing it into `NotAssessed` would render the most broken fit camdl
    /// can produce with the same neutral dash it uses for "not applicable".
    /// `NaN` is R's `NA`: the folded half was undefined for this marginal, so
    /// there genuinely is no band and the caller renders the reason instead.
    /// [`Stat::from_f64`] draws the same line.
    pub fn of(rhat: f64) -> Self {
        if rhat.is_nan() {
            Self::NotAssessed
        } else if rhat.is_infinite() {
            Self::Severe
        } else if rhat < RHAT_CONVERGED_THRESHOLD {
            Self::Converged
        } else if rhat < RHAT_SEVERE_THRESHOLD {
            Self::NotConverged
        } else {
            Self::Severe
        }
    }

    /// The plain glyph; colour is the renderer's business.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Converged => "✓",
            Self::NotConverged => "~",
            Self::Severe => "✗",
            Self::NotAssessed => "—",
        }
    }

    /// The verdict in words, so a glyph never has to carry the whole meaning
    /// on a surface a public-health decision is read off.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::NotConverged => "NOT converged",
            Self::Severe => "NOT converged — the chains are in different places",
            Self::NotAssessed => "not assessed",
        }
    }

    /// The glyph, never greener than a drawn finding allows.
    ///
    /// `drew_finding` is `R̂ > RHAT_REPORT_THRESHOLD` at the call site. The two
    /// thresholds are independent knobs, and if the finding bar is ever set
    /// BELOW the certification bar — gh#84's open 1.01 candidate would do
    /// exactly that — a parameter could print green beside an error-level
    /// finding drawn on the same number. Demoting here makes that
    /// unrepresentable rather than a coincidence of the current constants.
    pub fn glyph_with_finding(self, drew_finding: bool) -> &'static str {
        match (self, drew_finding) {
            (Self::Converged, true) => Self::NotConverged.glyph(),
            _ => self.glyph(),
        }
    }
}

/// The max-over-parameters R̂, or the reason there isn't one.
///
/// `max_rhat` previously folded from `0.0` over a map that a refused parameter
/// never entered, so a fit where every parameter was refused reported
/// `max R̂ = 0.000 ✓` and `converged: true` — a fit that could not be assessed
/// certifying itself. Reachable whenever the sampler never accepted a move
/// (every chain internally constant ⇒ R̂ non-finite) or a draw was `−inf`
/// (gh#607).
///
/// The three non-reporting arms are deliberately distinct because they call for
/// different actions, and because collapsing them is what produced the defect:
/// `Unassessable` means the sampler misbehaved and the fit is NOT converged;
/// `NotApplicable` means the run was never given the shape a between-chain
/// statistic needs, so the honest word is "not assessed"; `NoParams` means
/// there was nothing to assess. None of them is "converged".
#[derive(Debug, Clone, PartialEq)]
pub enum MaxRhat {
    /// Every assessed parameter reported, and this is the worst.
    Reported(f64),
    /// These chains took no step after burn-in (gh#1032): not converged,
    /// whatever any R̂ says. Ascending.
    ChainsFrozen { chains: Vec<ChainIndex> },
    /// At least one parameter's R̂ could not be computed for a reason that
    /// indicates a PROBLEM, or is unavailable with no reason recorded (a
    /// `null` in the stored summary, a refusal this build cannot read).
    /// `params` names them (ascending).
    Unassessable { params: Vec<String> },
    /// R̂ was never computable for a structural reason — too few chains, too
    /// few draws, unequal chain lengths. Report "not assessed".
    NotApplicable { reason: RhatRefusal },
    /// No parameter was assessed and no reason was recorded.
    NoParams,
}

/// The min-over-parameters ESS, or the reason there isn't one.
///
/// Bulk ESS is the rank-normalized cross-chain statistic of Vehtari et al.
/// (2021). It is never suppressed by R̂: it uses the between-chain variance
/// rather than summing per-chain estimates, so it does not overstate the
/// effective N when chains sit in different modes, and there is nothing to
/// gate. (The previous `ess_total` — a sum of per-chain Geyer estimates set to
/// NaN above R̂ 1.1 — was removed in gh#299 precisely because it reported
/// nothing exactly when a fit most needed a number.)
///
/// A parameter can still carry no bulk ESS: `rank_convergence` refused its
/// draws, or the rank-transformed draws are constant so the autocorrelation is
/// undefined. Those parameters reach `PosteriorDiagnostics` in one of two
/// encodings: **absent** on the loaded path (a NaN serializes to JSON `null`,
/// which `read_f64_map` drops) or **NaN-valued** on the `--exclude-chains`
/// recompute path (`chain_selection::recompute_subset_diagnostics`).
///
/// Either way, a minimum taken over the parameters that *did* report is a
/// minimum over a subset — and it RISES as a fit gets worse, because the
/// badly-mixing parameters leave the map and the well-mixing survivors set the
/// minimum. Measured (gh#687), two runs of one model differing only in particle
/// count: N=1200 gave max R̂ 2.639, min-param ESS 559, ESS/iter 0.013; N=4800
/// gave max R̂ 1.455, min-param ESS 73, ESS/iter 0.001. The better fit's
/// efficiency headline was 13x worse, purely because it converged four more
/// parameters into the map. So the minimum is reported only when every
/// parameter that could carry a pooled ESS carries one, and the blank is
/// rendered with the names of the ones that do not.
#[derive(Debug, Clone, PartialEq)]
pub enum MinEss {
    /// Every assessed parameter reports a pooled ESS; this is the slowest.
    Reported(f64),
    /// At least one assessed parameter has no pooled ESS. `missing` names them
    /// (ascending); `n_expected` is how many parameters were assessed, so a
    /// renderer can say "k of n".
    Unreportable {
        missing: Vec<String>,
        n_expected: usize,
    },
    /// No parameter was assessed across chains at all — nothing to minimize
    /// over (an empty diagnostics record, or a single-chain fit).
    NoParams,
}

/// The verdict on a method, decided by its R̂ alone (`verdict`). Serialized
/// with the report so a consumer reads it rather than recomputing it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    Converged { max_rhat: f64, param: Option<String> },
    NotConverged { max_rhat: f64, param: Option<String>, band: crate::convergence::RhatBand },
    /// These chains took no step after burn-in (gh#1032): not converged,
    /// whatever any R̂ says.
    ChainsFrozen { chains: Vec<crate::convergence::ChainIndex> },
    /// R̂ could not be computed for these parameters for a reason that is a
    /// problem: not converged, with no number to show.
    Unassessable { params: Vec<String> },
    /// R̂ was never computable for a structural reason (too few chains, a
    /// point start): not assessed.
    NotAssessed { reason: String },
    /// IF2, NLopt and pfilter: no chains to compare.
    NotApplicable,
}

/// A chain's number, counted from 0 — the same number in every artifact and
/// message that names a chain, and the `chain` key of `draws.tsv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChainIndex(pub usize);

/// The directory-name prefix: a chain's outputs live under `chain_<n>/`.
const DIR_PREFIX: &str = "chain_";

impl ChainIndex {
    /// The chain numbers `0 .. n`, in order.
    pub fn all(n: usize) -> impl Iterator<Item = ChainIndex> {
        (0..n).map(ChainIndex)
    }

    /// `chain_<n>`: the name of the directory this chain's outputs are under.
    pub fn dir_name(self) -> String {
        format!("{DIR_PREFIX}{}", self.0)
    }

    /// `<method_dir>/chain_<n>`.
    pub fn dir_in(self, method_dir: &Path) -> PathBuf {
        method_dir.join(self.dir_name())
    }

    /// The chain whose directory is named `name`, when `name` is `chain_<n>`.
    pub fn parse_dir_name(name: &str) -> Option<ChainIndex> {
        name.strip_prefix(DIR_PREFIX)?.parse().ok().map(ChainIndex)
    }
}

impl fmt::Display for ChainIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate so width and alignment flags (`{:>5}`) apply to the number.
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::str::FromStr for ChainIndex {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(ChainIndex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every refusal must survive the trip to `*_summary.json` and back with
    /// its numbers intact, or `camdl fit summary` cannot say more than an
    /// em-dash. `NonFiniteDraw` is the one that needs care: its `value` is
    /// non-finite by construction, and `serde_json` writes any non-finite
    /// `f64` as `null`, which then fails to deserialize as a number. It is
    /// carried as its `Display` form instead.
    #[test]
    fn every_refusal_round_trips_through_json_with_its_numbers() {
        let cases = [
            ConvergenceError::TooFewChains { n_chains: 1 },
            ConvergenceError::TooFewDraws { n_draws: 3 },
            ConvergenceError::UnequalChainLengths { expected: 50, chain: 1, found: 40 },
            ConvergenceError::NonFiniteDraw { chain: 1, draw: 7, value: f64::NEG_INFINITY },
            ConvergenceError::NonFiniteDraw { chain: 0, draw: 0, value: f64::INFINITY },
            ConvergenceError::ConstantDraws { value: 2.5 },
        ];
        for want in cases {
            let text = serde_json::to_string(&want).expect("serializes");
            assert!(!text.contains("null"),
                "a null loses the number the message is made of: {text}");
            let got: ConvergenceError =
                serde_json::from_str(&text).expect("round-trips");
            assert_eq!(got, want, "via {text}");
            // And the sentence a reader sees survives with it.
            assert_eq!(got.to_string(), want.to_string());
        }

        // NaN is not `PartialEq` with itself, so it is checked on the message.
        let nan = ConvergenceError::NonFiniteDraw { chain: 2, draw: 3, value: f64::NAN };
        let text = serde_json::to_string(&nan).expect("serializes");
        let got: ConvergenceError = serde_json::from_str(&text).expect("round-trips");
        assert!(matches!(got, ConvergenceError::NonFiniteDraw { chain: 2, draw: 3, value }
            if value.is_nan()), "got {got:?} via {text}");
    }

    /// The two halves of the headline, and what their order means. No cutoff
    /// on the gap is applied — only which is larger is reported.
    #[test]
    fn the_driver_names_the_larger_half_and_is_undefined_when_one_is() {
        assert_eq!(RhatDriver::of(1.42, 1.00), Some(RhatDriver::Location));
        assert_eq!(RhatDriver::of(1.00, 1.31), Some(RhatDriver::Scale));
        // Ties go to location: `max` took the bulk value, and there is no
        // spread disagreement to report.
        assert_eq!(RhatDriver::of(1.10, 1.10), Some(RhatDriver::Location));
        // An undefined folded half leaves nothing to compare.
        assert_eq!(RhatDriver::of(f64::INFINITY, f64::NAN), None);
        assert_eq!(RhatDriver::of(f64::NAN, 1.0), None);
        // A frozen pair: bulk is a well-defined +inf, so the comparison holds.
        assert_eq!(RhatDriver::of(f64::INFINITY, 1.0), Some(RhatDriver::Location));
    }

    /// `Infinite` and `Undefined` are different answers and must stay
    /// different **through serialization**.
    ///
    /// This is why `Stat` is a three-arm enum rather than `Option<f64>`:
    /// `serde_json` writes any non-finite `f64` as `null`, so `Some(INFINITY)`
    /// and `None` are the same bytes on disk. Collapsing them would turn "∞ —
    /// the sampler never moved" into "not computed" the moment a summary
    /// round-trips, which is the loudest signal camdl has silently becoming
    /// the quietest.
    #[test]
    fn infinite_and_undefined_are_distinct_through_a_json_round_trip() {
        assert_eq!(Stat::from_f64(f64::INFINITY), Stat::Infinite);
        assert_eq!(Stat::from_f64(f64::NEG_INFINITY), Stat::Infinite);
        assert_eq!(Stat::from_f64(f64::NAN), Stat::Undefined);
        assert_eq!(Stat::from_f64(1.03), Stat::Value(1.03));
        assert_ne!(Stat::Infinite, Stat::Undefined);

        for original in [Stat::Value(1.03), Stat::Infinite, Stat::Undefined] {
            let json = serde_json::to_string(&original).expect("serialize");
            let back: Stat = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, original, "round-trip must preserve the arm: {json}");
        }
        // The specific collapse the enum exists to prevent.
        assert_ne!(
            serde_json::to_string(&Stat::Infinite).unwrap(),
            serde_json::to_string(&Stat::Undefined).unwrap(),
            "the two must not serialize to the same bytes"
        );

        // And they render differently.
        assert_eq!(Stat::Infinite.cell(3, "—"), "∞");
        assert_eq!(Stat::Undefined.cell(3, "—"), "—");
        assert_eq!(Stat::Value(1.0295).cell(3, "—"), "1.030");
    }

    /// Every non-reporting state carries a reason a reader can act on. A blank
    /// cell with no explanation is what sent people to `diagnostics.json`.
    #[test]
    fn every_non_reporting_state_explains_itself() {
        let frozen = ParamConvergence::Scored {
            rhat: Stat::Infinite,
            rhat_bulk: Stat::Infinite, rhat_folded: Stat::Undefined,
            rhat_classic: Stat::Infinite,
            ess_bulk: Stat::Value(3.0), ess_tail: Stat::Undefined,
            ess_per_chain: Vec::new(), all_chains_frozen: true, rhat_withheld: None,
        };
        assert!(frozen.why_no_rhat().expect("frozen explains itself").contains("never accepted"));
        assert!(frozen.is_pathology());

        let undefined_fold = ParamConvergence::Scored {
            rhat: Stat::Undefined,
            rhat_bulk: Stat::Value(1.02), rhat_folded: Stat::Undefined,
            rhat_classic: Stat::Value(1.02),
            ess_bulk: Stat::Value(500.0), ess_tail: Stat::Value(400.0),
            ess_per_chain: Vec::new(), all_chains_frozen: false, rhat_withheld: None,
        };
        assert!(undefined_fold.why_no_rhat().expect("an undefined R̂ explains itself")
            .contains("folded"));

        let not_scored =
            ParamConvergence::NotScored { reason: RhatRefusal::NonFiniteDraw, detail: None };
        let why = not_scored.why_no_rhat().expect("a refusal explains itself");
        assert!(why.contains("NaN") || why.contains("infinite"), "got {why}");
        assert!(not_scored.is_pathology(), "a non-finite draw is a pathology");

        // Structural: not a pathology, but still not "converged".
        let too_few =
            ParamConvergence::NotScored { reason: RhatRefusal::TooFewChains, detail: None };
        assert!(!too_few.is_pathology(), "being given one chain is not a sampler failure");
        assert!(too_few.why_no_rhat().is_some(), "but it still says so");

        // An infinite R̂ is a pathology on its own, whether or not the
        // frozen-chains flag was set: R̂ = ∞ means the within-chain variance is
        // zero, and there is no benign way for that to happen.
        let inf_only = ParamConvergence::Scored {
            rhat: Stat::Infinite,
            rhat_bulk: Stat::Infinite, rhat_folded: Stat::Undefined,
            rhat_classic: Stat::Infinite,
            ess_bulk: Stat::Value(4.0), ess_tail: Stat::Undefined,
            ess_per_chain: Vec::new(), all_chains_frozen: false, rhat_withheld: None,
        };
        assert!(inf_only.is_pathology(),
            "R̂ = ∞ is a pathology by itself, not only when the flag agrees");

        // A healthy parameter has nothing to explain.
        let ok = ParamConvergence::Scored {
            rhat: Stat::Value(1.01),
            rhat_bulk: Stat::Value(1.01), rhat_folded: Stat::Value(1.00),
            rhat_classic: Stat::Value(1.00),
            ess_bulk: Stat::Value(900.0), ess_tail: Stat::Value(850.0),
            ess_per_chain: Vec::new(), all_chains_frozen: false, rhat_withheld: None,
        };
        assert_eq!(ok.why_no_rhat(), None);
        assert!(!ok.is_pathology());
    }

    /// The proposal pins `gate_verdict` strings to `pass` / `fail_a`
    /// / `fail_db` / `fail_both` (proposal §2). `serde_json` is the
    /// projection — assert the rendered scalar matches exactly so a
    /// future `rename_all` change to the enum doesn't silently shift
    /// the wire format.
    #[test]
    fn gate_verdict_serializes_to_proposal_strings() {
        for (variant, expected) in [
            (GateVerdict::Pass, "pass"),
            (GateVerdict::FailA, "fail_a"),
            (GateVerdict::FailDb, "fail_db"),
            (GateVerdict::FailBoth, "fail_both"),
            (GateVerdict::Incomplete, "incomplete"),
            (GateVerdict::NotAssessed, "not_assessed"),
        ] {
            let s = serde_json::to_value(variant).unwrap();
            assert_eq!(s, serde_json::Value::String(expected.into()));
        }
    }

    #[test]
    fn displays_the_bare_number_and_honours_width() {
        assert_eq!(ChainIndex(0).to_string(), "0");
        assert_eq!(ChainIndex(12).to_string(), "12");
        assert_eq!(format!("{:>4}|", ChainIndex(7)), "   7|");
        assert_eq!(format!("{:<3}|", ChainIndex(7)), "7  |");
    }

    #[test]
    fn directory_names_round_trip() {
        assert_eq!(ChainIndex(0).dir_name(), "chain_0");
        assert_eq!(ChainIndex(0).dir_in(Path::new("/leaf")), PathBuf::from("/leaf/chain_0"));
        assert_eq!(ChainIndex::parse_dir_name("chain_10"), Some(ChainIndex(10)));
        assert_eq!(ChainIndex::parse_dir_name("chain_"), None);
        assert_eq!(ChainIndex::parse_dir_name("chain_x"), None);
        assert_eq!(ChainIndex::parse_dir_name("draws.tsv"), None);
        assert_eq!(ChainIndex::parse_dir_name("chain_starts.tsv"), None);
    }

    #[test]
    fn parses_from_a_cli_token() {
        assert_eq!("0".parse::<ChainIndex>().unwrap(), ChainIndex(0));
        assert_eq!("5".parse::<ChainIndex>().unwrap(), ChainIndex(5));
        assert!("-1".parse::<ChainIndex>().is_err());
        assert!("x".parse::<ChainIndex>().is_err());
    }

    #[test]
    fn all_counts_from_zero() {
        assert_eq!(ChainIndex::all(3).collect::<Vec<_>>(),
            vec![ChainIndex(0), ChainIndex(1), ChainIndex(2)]);
        assert_eq!(ChainIndex::all(0).count(), 0);
    }
}
