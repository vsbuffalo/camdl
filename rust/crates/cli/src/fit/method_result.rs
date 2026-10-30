//! Typed loaded interpretation of a completed fit-stage.
//!
//! One variant per inference method: each carries the typed payload its
//! method produces, so consumers pattern-match instead of stringly-dispatching
//! on the stage leaf's `inputs.method` tag.
//!
//! Three variants — pfilter is excluded by design (it's a CLI
//! evaluator on already-fixed parameters, never a fit-stage). See
//! `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §2.
//!
//! Map fields use [`BTreeMap<String, _>`] end-to-end (never `HashMap`)
//! so `serde_json` produces lexicographically-ordered JSON output.
//! This is load-bearing for step 5's `summary ⊆ table` byte-equality
//! test (Deliverable C). A clippy lint or unit test guarding this
//! would be reasonable insurance — for now, the type definitions are
//! the authoritative constraint.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sim::inference::convergence::{ConvergenceError, RhatDriver, RhatRefusal};

use crate::chain_index::ChainIndex;
use crate::fit::state::FitState;

/// One stage of a fit, typed by method. The variant carries the
/// payload appropriate to its inference method (point estimate +
/// gates for IF2; posterior summaries + R̂ for Bayesian; status +
/// per-chain spread for NLopt deterministic MLE).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case")]
pub enum MethodResult {
    If2(If2StageResult),
    Pgas(PgasStageResult),
    Pmmh(PmmhStageResult),
    #[serde(rename = "nl-sbplx", alias = "nl-bobyqa", alias = "nl-lbfgs")]
    Nlopt(NloptStageResult),
}

/// Compound scout-convergence gate verdict (the IF2 "is this stage
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

/// IF2-stage result: point estimate + convergence diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct If2StageResult {
    pub best_loglik: f64,
    /// The winning chain as `fit_state.toml` stores it: the number of a
    /// `chain_N/` directory in the stage dir (gh#912, gh#990).
    pub best_chain: ChainIndex,
    /// Winner θ̂ from clean-eval. Estimated parameters only — fixed
    /// params live elsewhere (e.g. `final_params.toml` carries both).
    pub theta_hat: BTreeMap<String, f64>,
    /// Maximum chain-agreement Â over estimated params. **Â, not
    /// Gelman-Rubin R̂** — they're computed differently (Â is
    /// per-parameter chain-agreement on the IF2 trace's tail; R̂ is
    /// the standard MCMC convergence diagnostic). Renderers must not
    /// merge the two columns. See proposal §2. `None` when no Â was
    /// computed (one chain), rather than a 0.0 that reads as perfect.
    pub max_chain_agreement: Option<f64>,
    pub gate_verdict: GateVerdict,
    /// Particle-filter ESS evaluated at the clean-eval winner θ̂.
    /// `None` when the stage didn't compute one (e.g. clean-eval was
    /// disabled or the file is absent).
    pub ess_at_mle: Option<EssSummary>,
    pub n_chains: usize,
    pub n_iter: usize,
}

/// PF ESS summary at the IF2 winner θ̂. Three numbers; renderers can
/// pick whichever the table needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EssSummary {
    pub ess_min: f64,
    pub ess_mean: f64,
    /// Index of the observation step where ESS hit its minimum
    /// (1-based for human-readable display, matches
    /// `chain_evaluations.tsv` convention; `None` when ESS computation
    /// failed across the board).
    pub ess_min_step: Option<usize>,
}

/// Posterior convergence + efficiency diagnostics, shared by every Bayesian
/// sampler (PGAS, PMMH, mh). Every sampler computes these the same way
/// — per-param rank-normalized R̂ and bulk/tail ESS via
/// [`crate::fit::runner::compute_rhat_ess`]
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

/// Map fields use [`BTreeMap`] so any serialization is lexicographically
/// ordered (consistent with the rest of `method_result`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosteriorDiagnostics {
    /// One entry per **estimated** parameter — and the key set is exactly the
    /// estimated set, not "the parameters we managed to diagnose".
    ///
    /// That distinction is load-bearing. Parameter identity used to be derived
    /// from whichever diagnostic map happened to be non-empty, so a fit whose
    /// R̂ could not be computed reported *no parameters at all*: "we could not
    /// measure this" silently became "this does not exist". A parameter that
    /// could not be scored is present here as
    /// [`ParamConvergence::NotScored`] carrying its reason.
    pub per_param: BTreeMap<String, ParamConvergence>,
    /// Number of post-burn-in thinned posterior samples across all chains (as
    /// written to `draws.tsv`).
    pub n_samples: usize,
    /// Thinning factor (keep every Nth sweep). `n_samples × thin` = raw
    /// sampling iterations — the thinning-invariant denominator for
    /// ESS/iteration.
    #[serde(default = "default_thin")]
    pub thin: usize,
    /// Stage wall-clock (seconds), from `run.json inputs.wall_time_seconds`.
    /// `None` on older runs that predate the field. Denominator for ESS/second.
    #[serde(default)]
    pub wall_time_secs: Option<f64>,
    /// Number of chains. R̂ requires ≥2; part of "how this posterior was
    /// sampled", so it lives here rather than on each stage result.
    pub n_chains: usize,
}

impl PosteriorDiagnostics {
    /// Every scored parameter's R̂ becomes "not assessed: every chain started
    /// at one point", carrying the refusal with its numbers. The between-chain
    /// statistic over chains that began together agrees by construction, so
    /// the leaf's stored R̂ is not read as a verdict (proposal 2026-09-08,
    /// §3.4); the ESS is kept, and a parameter the estimator refused for its
    /// own reason keeps that reason. The R̂ numbers stay in the leaf's summary
    /// JSON for anyone who wants them.
    pub fn demote_point_start(&mut self, n_chains: usize, starts: &str) {
        let refusal = ConvergenceError::PointStart { n_chains, starts: starts.to_string() };
        for entry in self.per_param.values_mut() {
            if let ParamConvergence::Scored {
                rhat, rhat_bulk, rhat_folded, rhat_classic, all_chains_frozen, rhat_withheld, ..
            } = entry
            {
                *rhat = Stat::Undefined;
                *rhat_bulk = Stat::Undefined;
                *rhat_folded = Stat::Undefined;
                *rhat_classic = Stat::Undefined;
                *all_chains_frozen = false;
                *rhat_withheld = Some(refusal.clone());
            }
        }
    }

    /// The worst R̂ over the assessed params — **or the reason there isn't
    /// one**. The convergence headline; **R̂, not IF2's Â**. See [`MaxRhat`].
    pub fn max_rhat_status(&self) -> MaxRhat {
        if self.per_param.is_empty() {
            return MaxRhat::NoParams;
        }
        // A parameter whose failure is a SAMPLER pathology sinks the verdict,
        // whether or not other parameters reported. A fit is not converged
        // because some of it was. So does a parameter the estimator scored
        // whose R̂ is unavailable with no reason recorded — a `null` in the
        // summary, or a refusal this build cannot read: skipping it would
        // pass the fit on the parameters that happened to report.
        let unavailable = |p: &ParamConvergence| matches!(p,
            ParamConvergence::Scored {
                rhat: Stat::Undefined, rhat_withheld: None, all_chains_frozen: false, ..
            });
        let bad: Vec<String> = self
            .per_param
            .iter()
            .filter(|(_, p)| p.is_pathology() || unavailable(p))
            .map(|(n, _)| n.clone())
            .collect();
        if !bad.is_empty() {
            return MaxRhat::Unassessable { params: bad };
        }
        // An R̂ withheld for how the chains were started is not a number and
        // not a pathology: the question was never answerable here.
        if let Some(reason) = self.per_param.values().find_map(|p| match p {
            ParamConvergence::Scored { rhat_withheld: Some(e), .. } => Some(e.refusal()),
            _ => None,
        }) {
            return MaxRhat::NotApplicable { reason };
        }
        let mut worst = f64::NEG_INFINITY;
        let mut any = false;
        for p in self.per_param.values() {
            if let Stat::Value(v) = p.rhat() {
                worst = worst.max(v);
                any = true;
            }
        }
        if any {
            return MaxRhat::Reported(worst);
        }
        // Nothing reported and nothing pathological: the run was never given
        // the shape a between-chain statistic needs.
        let reason = self
            .per_param
            .values()
            .find_map(|p| match p {
                ParamConvergence::NotScored { reason, .. } => Some(*reason),
                _ => None,
            })
            .unwrap_or(RhatRefusal::EstimatedSetUnknown);
        MaxRhat::NotApplicable { reason }
    }

    /// The worst R̂ as a number, `None` when there isn't one. Thin projection of
    /// [`max_rhat_status`](Self::max_rhat_status) for the display sites that
    /// only need the value.
    pub fn max_rhat(&self) -> Option<f64> {
        match self.max_rhat_status() {
            MaxRhat::Reported(v) => Some(v),
            _ => None,
        }
    }

    /// Whether this fit may be reported as converged. **Only** `Reported`
    /// below the threshold qualifies: `Unassessable` is a sampler pathology,
    /// and `NotApplicable` / `NoParams` mean the question was never answered.
    pub fn converged_at(&self, threshold: f64) -> bool {
        matches!(self.max_rhat_status(), MaxRhat::Reported(v) if v < threshold)
    }

    /// One parameter's R̂ as a table cell.
    ///
    /// A fit that fails says so once, in the headline. WHICH parameter failed
    /// is only readable here, and a column of dashes over values the run
    /// already computed sends the reader to `diagnostics.json` to answer the
    /// first question they have (gh#611).
    pub fn rhat_cell(&self, name: &str, missing: &str) -> String {
        self.per_param
            .get(name)
            .map(|p| p.rhat().cell(3, missing))
            .unwrap_or_else(|| missing.to_string())
    }

    /// One parameter's bulk ESS as a table cell.
    pub fn ess_cell(&self, name: &str, missing: &str) -> String {
        self.per_param
            .get(name)
            .map(|p| p.ess_bulk().cell(0, missing))
            .unwrap_or_else(|| missing.to_string())
    }

    /// One parameter's tail ESS as a table cell. `Undefined` for a parameter
    /// piled on a bound, whose tail indicator is constant.
    pub fn ess_tail_cell(&self, name: &str, missing: &str) -> String {
        self.per_param
            .get(name)
            .map(|p| p.ess_tail().cell(0, missing))
            .unwrap_or_else(|| missing.to_string())
    }

    /// Minimum bulk ESS over the assessed params — the slowest param bounds
    /// the usable ESS — **or the reason there isn't one**. See [`MinEss`].
    pub fn min_ess_status(&self) -> MinEss {
        if self.per_param.is_empty() {
            return MinEss::NoParams;
        }
        let n_expected = self.per_param.len();
        let mut missing: Vec<String> = Vec::new();
        let mut min = f64::INFINITY;
        for (name, p) in &self.per_param {
            match p.ess_bulk() {
                Stat::Value(v) => min = min.min(v),
                _ => missing.push(name.clone()),
            }
        }
        if missing.is_empty() {
            MinEss::Reported(min)
        } else {
            MinEss::Unreportable { missing, n_expected }
        }
    }

    /// The min-param ESS as a number, `None` when it is not reportable.
    pub fn min_ess(&self) -> Option<f64> {
        match self.min_ess_status() {
            MinEss::Reported(v) => Some(v),
            MinEss::Unreportable { .. } | MinEss::NoParams => None,
        }
    }

    /// Bulk ESS per param, for the consumers that want the raw map (the
    /// `fit table` row's `ess_posterior`). Non-finite entries are omitted —
    /// a JSON consumer reading this cannot represent them anyway.
    pub fn ess_per_param(&self) -> BTreeMap<String, f64> {
        self.per_param
            .iter()
            .filter_map(|(n, p)| p.ess_bulk().finite().map(|v| (n.clone(), v)))
            .collect()
    }

    /// Raw sampling iterations = `n_samples × thin`. Recovers the raw sampling
    /// steps from the kept (thinned) draws, making the two efficiency metrics
    /// invariant to thinning + iteration count.
    pub fn raw_iters(&self) -> usize {
        self.n_samples.saturating_mul(self.thin.max(1))
    }

    /// ESS per raw sampling iteration = `min_ess / (n_samples × thin)`. The
    /// **algorithm-quality** metric: hardware-independent, the number to
    /// compare samplers with. `None` when there are no samples/params.
    pub fn ess_per_iter(&self) -> Option<f64> {
        let raw = self.raw_iters();
        if raw == 0 {
            return None;
        }
        Some(self.min_ess()? / raw as f64)
    }

    /// ESS per wall-clock second = `min_ess / wall_time`. The **runtime**
    /// metric: thinning-invariant but hardware-dependent (runtime-to-target).
    /// `None` when wall-time is absent/zero or there are no params.
    pub fn ess_per_sec(&self) -> Option<f64> {
        let secs = self.wall_time_secs.filter(|s| *s > 0.0)?;
        Some(self.min_ess()? / secs)
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

/// The R̂ above which a surface marks a parameter ✗ rather than ~ — "not
/// converged" against "badly enough that the chains are somewhere else".
/// Display only: nothing keys on it, and it draws no diagnostic of its own.
pub const RHAT_SEVERE_THRESHOLD: f64 = 1.5;

/// Where a reported R̂ sits against the bar camdl **certifies** against.
///
/// One band for every surface that glyphs an R̂. Two surfaces used to spell it
/// differently — the end-of-stage block against `RHAT_REPORT_THRESHOLD` (1.1,
/// which is the bar that draws a *finding*), `fit summary` against
/// `RHAT_CONVERGED_THRESHOLD` (1.05, the bar `converged_at` and the
/// machine-readable `converged` column key on). A parameter in `[1.05, 1.1)`
/// therefore printed ✓ when the fit finished and ✗ when the user ran
/// `fit summary` on the same directory.
///
/// Certification and severity are separate questions and stay separate: this
/// band answers "may this be reported as converged", and
/// `DiagnosticKind::severity` answers "is the finding a warning or an error".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// undefined. Those parameters reach [`PosteriorDiagnostics`] in one of two
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

/// PGAS-stage result: posterior approximation. Convergence + efficiency
/// diagnostics live in the shared [`PosteriorDiagnostics`]; the fields here are
/// PGAS-specific (posterior moments + per-param acceptance from its inner
/// Gibbs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PgasStageResult {
    pub diagnostics: PosteriorDiagnostics,
    pub posterior_mean: BTreeMap<String, f64>,
    pub posterior_q025: BTreeMap<String, f64>,
    pub posterior_q975: BTreeMap<String, f64>,
    pub acceptance_per_param: BTreeMap<String, f64>,
}

/// Default thinning factor (1 = unthinned) for older runs whose summary JSON
/// predates the `thin` field.
fn default_thin() -> usize {
    1
}

/// NLopt-stage result: deterministic MLE on the ODE skeleton (Phase 1
/// of the ODE-inference proposal). Mirrors `If2StageResult`'s point-
/// estimate shape, with NLopt-specific convergence info (status and
/// per-param chain spread) replacing IF2's chain-agreement Â and
/// PF-ESS columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NloptStageResult {
    pub best_loglik: f64,
    /// The winning chain as `fit_state.toml` stores it, matching the `chain`
    /// column of `chain_results.tsv` beside it (gh#912, gh#990).
    pub best_chain: ChainIndex,
    /// Winner θ̂ — estimated parameters only. Fixed params live in
    /// `mle_params.toml` (full vector with provenance).
    pub theta_hat: BTreeMap<String, f64>,
    /// Which NLopt algorithm produced this result: `"nl-sbplx"`,
    /// `"nl-bobyqa"` or `"nl-lbfgs"`.
    pub algorithm: String,
    /// Number of converged (`Success` / `XtolReached` / `FtolReached`)
    /// chains. `n_chains - n_converged` includes both `MaxEvalReached`
    /// (soft failure) and `Failed` (hard error).
    pub n_converged: usize,
    /// Maximum per-param relative range across chains (range / bound
    /// width). NA when n_chains == 1.
    pub max_rel_range: f64,
    pub n_chains: usize,
    /// The optimizer's recorded verdict (`fit_state.toml`
    /// `[optimizer_convergence]`); `None` on a leaf written before it was
    /// recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_convergence: Option<crate::fit::state::OptimizerConvergence>,
}

/// PMMH-stage result: posterior approximation + scalar acceptance. Convergence
/// + efficiency diagnostics live in the shared [`PosteriorDiagnostics`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmmhStageResult {
    pub diagnostics: PosteriorDiagnostics,
    pub posterior_mean: BTreeMap<String, f64>,
    /// Scalar across chains (mean of per-chain rates). PGAS reports
    /// per-parameter rates because its inner Gibbs proposes parameters
    /// one at a time; PMMH proposes the full vector each step.
    pub acceptance_rate: f64,
    pub map_loglik: f64,
}

/// Errors loading a `MethodResult` from a stage directory.
#[derive(Debug)]
pub enum MethodResultError {
    /// `run.json` named a method this ADT doesn't carry. New methods
    /// get added by extending the enum, which produces compile errors
    /// at every consumer that doesn't handle them yet — exactly what
    /// this typing was designed to surface.
    UnknownMethod {
        method: String,
        stage_dir: PathBuf,
    },
    /// A required artifact was missing or unreadable.
    Io {
        stage_dir: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for MethodResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MethodResultError::UnknownMethod { method, stage_dir } => write!(
                f,
                "unknown fit-stage method `{}` at {} (expected if2, pgas, \
                 pmmh, mh, nl-sbplx, nl-bobyqa, or nl-lbfgs)",
                method,
                stage_dir.display()
            ),
            MethodResultError::Io { stage_dir, message } => {
                write!(f, "loading {}: {}", stage_dir.display(), message)
            }
        }
    }
}

impl std::error::Error for MethodResultError {}

impl MethodResult {
    /// Whether [`MethodResult::load_from`] has a variant for this algorithm.
    ///
    /// `pfilter` does not: it evaluates a likelihood at fixed θ with
    /// replicates rather than fitting, and writes no `fit_state.toml`. A
    /// caller choosing *which* leaf of a segment to summarize must skip such a
    /// leaf rather than pick it and fail — otherwise its mere presence drops
    /// the whole fit from `fit table` (gh#590). The match is exhaustive so a
    /// new algorithm must decide here; `has_result_agrees_with_load_from`
    /// pins it against `load_from`'s arms.
    pub fn has_result(algo: crate::run_meta::FitAlgorithm) -> bool {
        use crate::run_meta::FitAlgorithm as A;
        match algo {
            A::If2 | A::Pgas | A::Pmmh | A::Mh
            | A::NlSbplx | A::NlBobyqa | A::NlLbfgs => true,
            A::Pfilter => false,
        }
    }

    /// Dispatch on `method` and load the matching variant. Errors on
    /// unknown methods rather than silently producing a generic shape.
    pub fn load_from(stage_dir: &Path, method: &str) -> Result<Self, MethodResultError> {
        match method {
            "if2" => Ok(MethodResult::If2(If2StageResult::load(stage_dir)?)),
            "pgas" => Ok(MethodResult::Pgas(PgasStageResult::load(stage_dir)?)),
            // `mh` (deterministic Metropolis-Hastings) runs through the PMMH
            // runner and writes the same posterior artifacts (`fit_state.toml`,
            // `draws.tsv`, acceptance + R̂), so it loads as a Pmmh result. Without
            // this arm an `mh` fit was dropped from `fit table` / `fit summary`
            // entirely ("unknown fit-stage method `mh`") — a per-method allowlist
            // that excluded a method the rest of the system supports.
            "pmmh" | "mh" => {
                // mh writes its OWN `mh_summary.json` (not `pmmh_summary.json`)
                // though it shares the Pmmh result shape.
                let algo = if method == "mh" {
                    crate::run_meta::FitAlgorithm::Mh
                } else {
                    crate::run_meta::FitAlgorithm::Pmmh
                };
                Ok(MethodResult::Pmmh(PmmhStageResult::load(stage_dir, algo)?))
            }
            "nl-sbplx" | "nl-bobyqa" | "nl-lbfgs" => Ok(MethodResult::Nlopt(
                NloptStageResult::load(stage_dir, method)?,
            )),
            unknown => Err(MethodResultError::UnknownMethod {
                method: unknown.to_string(),
                stage_dir: stage_dir.to_owned(),
            }),
        }
    }
}

/// Read the stage leaf's `inputs` map from its `run.json` (`runid::RunRecord`).
/// Convenience for loaders that need stage-level metadata (`n_chains`,
/// `algorithm`) recorded alongside the leaf. Errors with a typed
/// [`MethodResultError::Io`] on a missing / malformed file or a non-`FitStage`
/// record — the contract is "every fit-stage writes a `RunRecord` leaf whose
/// `inputs` carries its config".
fn read_stage_inputs(stage_dir: &Path) -> Result<serde_json::Value, MethodResultError> {
    let bytes = std::fs::read(stage_dir.join("run.json")).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("run.json: {}", e),
    })?;
    let rec: runid::RunRecord = serde_json::from_slice(&bytes).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("run.json: {}", e),
    })?;
    if rec.kind != runid::ArtifactKind::FitStage {
        return Err(MethodResultError::Io {
            stage_dir: stage_dir.to_owned(),
            message: "run.json is not a fit-stage".into(),
        });
    }
    Ok(rec.inputs)
}

/// `inputs.n_chains` as a `usize` (0 when absent).
fn inputs_n_chains(inputs: &serde_json::Value) -> usize {
    inputs.get("n_chains").and_then(|v| v.as_u64()).unwrap_or(0) as usize
}

/// A multi-chain leaf whose `fit_state.toml` says every chain started at one
/// point, with the rule as written. `Ok(None)` for a spread start, one chain,
/// a leaf written before `chain_starts_kind` existed, or no `fit_state.toml`
/// at all — a missing record never demotes, because the demotion is a claim
/// about how the chains were started and an absent field makes no claim. A
/// `fit_state.toml` that exists but cannot be read is an error: reading it as
/// "no claim" would leave a point-started R̂ looking assessed.
pub fn point_start(stage_dir: &Path) -> Result<Option<(usize, String)>, MethodResultError> {
    if !stage_dir.join("fit_state.toml").is_file() {
        return Ok(None);
    }
    let state = FitState::load(&stage_dir.to_string_lossy()).map_err(|e| {
        MethodResultError::Io {
            stage_dir: stage_dir.to_owned(),
            message: format!("fit_state.toml: {e}"),
        }
    })?;
    if state.chain_starts_kind != Some(crate::fit::starts::ChainStartsKind::Point) {
        return Ok(None);
    }
    if state.n_chains < 2 {
        return Ok(None);
    }
    Ok(Some((
        state.n_chains,
        state.chain_init_source.unwrap_or_else(|| "a point rule".to_string()),
    )))
}

/// The one place a sampler leaf's diagnostics are read into a
/// [`PosteriorDiagnostics`], so the point-start demotion cannot be missed by
/// a loader that forgot it.
fn posterior_diagnostics(
    stage_dir: &Path,
    per_param: BTreeMap<String, ParamConvergence>,
    n_samples: usize,
    thin: usize,
    wall_time_secs: Option<f64>,
    n_chains: usize,
) -> Result<PosteriorDiagnostics, MethodResultError> {
    let mut diag = PosteriorDiagnostics { per_param, n_samples, thin, wall_time_secs, n_chains };
    if let Some((n, starts)) = point_start(stage_dir)? {
        diag.demote_point_start(n, &starts);
    }
    Ok(diag)
}

/// θ̂ for an optimizer leaf: its stored point (`start_values`, which for IF2
/// also holds the fixed parameters) restricted to the estimated set. That set
/// is `tail_chain_agreement`'s keys when the run computed Â (one per estimated
/// parameter), else the fit's own record of it (`fit.meta.json` `estimated`,
/// in the nearest ancestor segment). Only a leaf with neither — written
/// before either existed — falls back to every stored value.
fn theta_hat_from_state(stage_dir: &Path, state: &FitState) -> BTreeMap<String, f64> {
    let estimated = estimated_names(Some(stage_dir), state);
    state
        .start_values
        .iter()
        .filter(|(name, _)| estimated.as_ref().is_none_or(|set| set.contains(*name)))
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

/// The estimated parameters of an optimizer leaf, for [`theta_hat_from_state`]
/// and `fit summary`'s parameter listings. `None` only when the leaf records
/// neither Â nor (with `stage_dir`) a `fit.meta.json` `estimated` list.
pub(crate) fn estimated_names(
    stage_dir: Option<&Path>,
    state: &FitState,
) -> Option<std::collections::BTreeSet<String>> {
    if !state.tail_chain_agreement.is_empty() {
        Some(state.tail_chain_agreement.keys().cloned().collect())
    } else {
        stage_dir?
            .ancestors()
            .skip(1)
            .take(4)
            .find_map(crate::run_meta::read_fit_sidecar)
            // A sidecar written before `estimated` was recorded carries an
            // empty list: no record, not "nothing estimated".
            .filter(|side| !side.estimated.is_empty())
            .map(|side| side.estimated.into_iter().collect())
    }
}

// ── NloptStageResult ────────────────────────────────────────────────

impl NloptStageResult {
    /// The chain-agreement measurement (largest relative range across the
    /// converged chains), or `None` when the leg was not computed: fewer than
    /// two converged chains, where the stored 0.0 is a placeholder.
    pub fn chain_agreement(&self) -> Option<f64> {
        let compared = self.optimizer_convergence.as_ref()
            .map_or(self.n_chains, |v| v.n_converged);
        (compared >= 2).then_some(self.max_rel_range)
    }

    pub fn load(stage_dir: &Path, method: &str) -> Result<Self, MethodResultError> {
        let n_chains = inputs_n_chains(&read_stage_inputs(stage_dir)?);
        let state = FitState::load(&stage_dir.to_string_lossy()).map_err(|e| {
            MethodResultError::Io {
                stage_dir: stage_dir.to_owned(),
                message: format!("fit_state.toml: {}", e),
            }
        })?;

        // theta_hat: estimated parameters only. NLopt's tail_chain_agreement
        // (per-param relative range) carries one entry per estimated param
        // — same trick the IF2 loader uses to filter to the estimated set.
        let theta_hat = theta_hat_from_state(stage_dir, &state);

        let max_rel_range = state
            .tail_chain_agreement
            .values()
            .copied()
            .fold(0.0_f64, f64::max);

        Ok(NloptStageResult {
            best_loglik: state.best_loglik,
            best_chain: state.best_chain,
            theta_hat,
            algorithm: method.to_string(),
            n_converged: state.n_good_chains.unwrap_or(n_chains),
            max_rel_range,
            n_chains,
            optimizer_convergence: state.optimizer_convergence.clone(),
        })
    }
}

// ── If2StageResult ──────────────────────────────────────────────────

impl If2StageResult {
    pub fn load(stage_dir: &Path) -> Result<Self, MethodResultError> {
        let inputs = read_stage_inputs(stage_dir)?;
        let n_chains = inputs_n_chains(&inputs);

        let state = FitState::load(&stage_dir.to_string_lossy()).map_err(|e| {
            MethodResultError::Io {
                stage_dir: stage_dir.to_owned(),
                message: format!("fit_state.toml: {}", e),
            }
        })?;

        // theta_hat: estimated parameters only. We don't have direct
        // access to the [estimate] block from this side of the
        // pipeline, but `tail_chain_agreement` only contains estimated
        // params — we use that as the key set. When tail_chain_agreement
        // is empty (legacy file), fall back to the full start_values
        // map (which over-includes fixed params, but is the best we
        // can do without re-reading fit.toml).
        let theta_hat = theta_hat_from_state(stage_dir, &state);

        let max_chain_agreement = (!state.tail_chain_agreement.is_empty()).then(|| {
            state.tail_chain_agreement.values().copied().fold(0.0_f64, f64::max)
        });

        let gate_verdict = compute_if2_gate_verdict(&state);

        let ess_at_mle = read_ess_at_mle(stage_dir);

        let n_iter = inputs
            .get("algorithm")
            .and_then(|a| a.get("iterations"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        Ok(If2StageResult {
            best_loglik: state.best_loglik,
            best_chain: state.best_chain,
            theta_hat,
            max_chain_agreement,
            gate_verdict,
            ess_at_mle,
            n_chains,
            n_iter,
        })
    }
}

/// The compound IF2 gate on the persisted FitState, as a typed verdict:
/// [`crate::fit::gating::assess_if2`], the one measurement the source check
/// and `fit summary` also read. Falls back to `GateConfig::default()` when
/// `state.resolved_gate` is absent (legacy file pre-Phase-3). The decibans
/// leg is inconclusive when the leaf has no clean-eval data, and the verdict
/// is then judged on Â alone.
fn compute_if2_gate_verdict(state: &FitState) -> GateVerdict {
    let gate = state.resolved_gate.clone().unwrap_or_default();
    let assessed = crate::fit::gating::assess_if2(state, &gate);
    if assessed.is_not_assessed() {
        return GateVerdict::NotAssessed;
    }
    if assessed.is_incomplete() {
        // A failed decibans leg is still a failure; otherwise incomplete.
        return if assessed.spread.is_some_and(|leg| !leg.passes()) {
            GateVerdict::FailDb
        } else {
            GateVerdict::Incomplete
        };
    }
    match (assessed.a_passes(), assessed.spread.map(|leg| leg.passes())) {
        (true, Some(true)) | (true, None) => GateVerdict::Pass,
        (false, Some(false)) => GateVerdict::FailBoth,
        (false, _) => GateVerdict::FailA,
        (true, Some(false)) => GateVerdict::FailDb,
    }
}

/// Read ESS-at-MLE from `chain_evaluations.tsv` if present. The TSV
/// schema (set in `runner::write_clean_eval_tsv`):
/// `chain candidate loglik se ess_mean ess_min ess_min_step n_neg_inf_incr <param₁> ...`
///
/// We pick the row corresponding to the overall winner (max loglik
/// across all rows), since clean-eval re-scoring is done per
/// (chain × candidate) and the IF2's `best_chain` is the overall
/// max. Returns `None` if the file is absent or malformed.
fn read_ess_at_mle(stage_dir: &Path) -> Option<EssSummary> {
    let path = stage_dir.join("chain_evaluations.tsv");
    let contents = std::fs::read_to_string(&path).ok()?;
    let mut header: Option<Vec<String>> = None;
    let mut best_row: Option<(f64, EssSummary)> = None;
    for line in contents.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if header.is_none() {
            header = Some(cols.iter().map(|s| (*s).to_string()).collect());
            continue;
        }
        let h = header.as_ref().unwrap();
        let idx = |name: &str| -> Option<usize> { h.iter().position(|c| c == name) };
        let loglik = idx("loglik").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_mean = idx("ess_mean").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_min = idx("ess_min").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_min_step = idx("ess_min_step")
            .and_then(|i| cols.get(i))
            .and_then(|s| s.parse::<i64>().ok())
            .map(|i| if i < 0 { None } else { Some(i as usize) })
            .unwrap_or(None);
        let summary = EssSummary {
            ess_min,
            ess_mean,
            ess_min_step,
        };
        match &best_row {
            Some((best_ll, _)) if loglik <= *best_ll => {}
            _ => best_row = Some((loglik, summary)),
        }
    }
    best_row.map(|(_, s)| s)
}

// ── PgasStageResult ─────────────────────────────────────────────────

impl PgasStageResult {
    pub fn load(stage_dir: &Path) -> Result<Self, MethodResultError> {
        let inputs = read_stage_inputs(stage_dir)?;
        let n_chains = inputs_n_chains(&inputs);
        let wall_time_secs = inputs.get("wall_time_seconds").and_then(|v| v.as_f64());

        // Estimated parameter names from algorithm config — pgas
        // doesn't store them directly. Fall back to "every column in
        // draws.tsv" if not in algorithm.
        let summary =
            read_summary_json(stage_dir, &crate::run_meta::FitAlgorithm::Pgas.summary_filename())?;
        let thin = summary.get("thin").and_then(|v| v.as_u64()).map(|t| t as usize).unwrap_or(1);

        let conv = ConvergenceMaps::read(&summary);

        // Posterior moments: average each estimated-param column in
        // draws.tsv. The estimated-param key set is rhat_map's keys
        // when present (rhat is per estimated param), else ess_map's.
        // Parameter identity is the UNION of every diagnostic key plus every
        // recorded refusal — never "whichever map happened to be non-empty".
        // A fit whose R̂ could not be computed still has parameters (gh#611 /
        // review blocker 1); deriving the list from a diagnostic map made such
        // a fit report none at all.
        let est_names: Vec<String> = conv.param_names();
        let (n_samples, posterior_mean, posterior_q025, posterior_q975) =
            posterior_summaries(stage_dir, &est_names);

        // Acceptance per param: pgas writes acceptance_rates as
        // Vec<Vec<f64>> (n_chains × n_estimated). Aggregate to per-
        // param mean across chains.
        let acceptance_per_param: BTreeMap<String, f64> = {
            let raw = summary
                .get("acceptance_rates")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let n_chains = raw.len();
            if n_chains == 0 || est_names.is_empty() {
                BTreeMap::new()
            } else {
                let mut acc = vec![0.0_f64; est_names.len()];
                let mut counts = vec![0usize; est_names.len()];
                for chain in &raw {
                    let row = chain.as_array().cloned().unwrap_or_default();
                    for (i, v) in row.iter().enumerate() {
                        if i >= acc.len() {
                            break;
                        }
                        if let Some(x) = v.as_f64() {
                            acc[i] += x;
                            counts[i] += 1;
                        }
                    }
                }
                est_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let denom = counts[i].max(1) as f64;
                        (name.clone(), acc[i] / denom)
                    })
                    .collect()
            }
        };

        Ok(PgasStageResult {
            diagnostics: posterior_diagnostics(
                stage_dir, conv.per_param(), n_samples, thin, wall_time_secs, n_chains,
            )?,
            posterior_mean,
            posterior_q025,
            posterior_q975,
            acceptance_per_param,
        })
    }
}

// ── PmmhStageResult ─────────────────────────────────────────────────

impl PmmhStageResult {
    pub fn load(
        stage_dir: &Path,
        algo: crate::run_meta::FitAlgorithm,
    ) -> Result<Self, MethodResultError> {
        let inputs = read_stage_inputs(stage_dir)?;
        let n_chains = inputs_n_chains(&inputs);
        let wall_time_secs = inputs.get("wall_time_seconds").and_then(|v| v.as_f64());
        let summary = read_summary_json(stage_dir, &algo.summary_filename())?;
        let thin = summary.get("thin").and_then(|v| v.as_u64()).map(|t| t as usize).unwrap_or(1);

        let conv = ConvergenceMaps::read(&summary);
        // Parameter identity is the UNION of every diagnostic key plus every
        // recorded refusal — never "whichever map happened to be non-empty".
        // A fit whose R̂ could not be computed still has parameters (gh#611 /
        // review blocker 1); deriving the list from a diagnostic map made such
        // a fit report none at all.
        let est_names: Vec<String> = conv.param_names();
        let (n_samples, posterior_mean, _q025, _q975) =
            posterior_summaries(stage_dir, &est_names);

        // PMMH writes acceptance_rate as Vec<f64> (one per chain). The
        // table-row scalar is the mean across chains.
        let acceptance_rate = summary
            .get("acceptance_rate")
            .and_then(|v| v.as_array())
            .map(|a| {
                let v: Vec<f64> = a.iter().filter_map(|x| x.as_f64()).collect();
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .unwrap_or(0.0);
        let map_loglik = summary
            .get("map_loglik")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);

        Ok(PmmhStageResult {
            diagnostics: posterior_diagnostics(
                stage_dir, conv.per_param(), n_samples, thin, wall_time_secs, n_chains,
            )?,
            posterior_mean,
            acceptance_rate,
            map_loglik,
        })
    }
}

// ── shared helpers ──────────────────────────────────────────────────

/// The convergence half of a stage summary, read once.
///
/// The mirror of
/// [`StageConvergence::summary_fields`](crate::fit::runner::StageConvergence::summary_fields):
/// one writer, one reader, so a statistic the samplers store cannot be dropped
/// by a loader that forgot to add a key. Every field is optional on the wire —
/// a fit written before a statistic existed simply has no entry for it, and the
/// corresponding [`Stat`] is `Undefined` rather than absent.
#[derive(Debug, Default)]
pub struct ConvergenceMaps {
    rhat: BTreeMap<String, f64>,
    rhat_bulk: BTreeMap<String, f64>,
    rhat_folded: BTreeMap<String, f64>,
    rhat_classic: BTreeMap<String, f64>,
    refused: BTreeMap<String, RhatRefusal>,
    refusal_detail: BTreeMap<String, ConvergenceError>,
    ess: BTreeMap<String, f64>,
    ess_tail: BTreeMap<String, f64>,
    ess_per_chain: BTreeMap<String, Vec<f64>>,
    /// Parameters whose `rhat_not_reported` entry this build cannot read (a
    /// refusal of a newer camdl). Kept so the parameter cannot vanish; its R̂
    /// is unavailable.
    unreadable_refusals: std::collections::BTreeSet<String>,
}

impl ConvergenceMaps {
    /// Read every convergence key a stage summary may carry.
    pub fn read(summary: &serde_json::Value) -> Self {
        ConvergenceMaps {
            // A `null` R̂ (a NaN, as serde_json writes one) is kept as NaN:
            // the parameter is present with an unavailable R̂, never absent.
            rhat: read_f64_map_keeping_nan(summary, "rhat"),
            rhat_bulk: read_f64_map(summary, "rhat_bulk"),
            rhat_folded: read_f64_map(summary, "rhat_folded"),
            rhat_classic: read_f64_map(summary, "rhat_classic"),
            refused: read_typed_map(summary, "rhat_not_reported"),
            refusal_detail: read_typed_map(summary, "rhat_refusal_detail"),
            ess: read_f64_map(summary, "ess"),
            ess_tail: read_f64_map(summary, "ess_tail"),
            ess_per_chain: read_typed_map(summary, "ess_per_chain"),
            unreadable_refusals: unreadable_keys::<RhatRefusal>(summary, "rhat_not_reported"),
        }
    }

    /// Every parameter this summary mentions, in any capacity. The key set is
    /// the UNION, so a parameter that appears only as a refusal is still a
    /// parameter of this fit (gh#611).
    pub fn param_names(&self) -> Vec<String> {
        let mut set: std::collections::BTreeSet<&String> = Default::default();
        set.extend(self.rhat.keys());
        set.extend(self.rhat_bulk.keys());
        set.extend(self.rhat_folded.keys());
        set.extend(self.rhat_classic.keys());
        set.extend(self.refused.keys());
        set.extend(self.refusal_detail.keys());
        set.extend(self.ess.keys());
        set.extend(self.ess_tail.keys());
        set.extend(self.ess_per_chain.keys());
        set.extend(self.unreadable_refusals.iter());
        set.into_iter().cloned().collect()
    }

    /// Assemble the per-parameter sum type.
    ///
    /// The loaded path is just another producer of that type — a band label and
    /// `fit summary` must reduce one fit the same way, whether the numbers came
    /// from a recompute or off disk.
    pub fn per_param(&self) -> BTreeMap<String, ParamConvergence> {
        let stat = |m: &BTreeMap<String, f64>, name: &str| {
            m.get(name).copied().map(Stat::from_f64).unwrap_or(Stat::Undefined)
        };
        self.param_names()
            .into_iter()
            .map(|name| {
                // A refusal with no numbers at all is `NotScored`; a refusal
                // alongside an ESS means the estimator ran and only R̂ is missing.
                let reason = self.refused.get(&name);
                let entry = match (reason, self.ess.get(&name)) {
                    (Some(reason), None) => ParamConvergence::NotScored {
                        reason: *reason,
                        detail: self.refusal_detail.get(&name).cloned(),
                    },
                    (reason, _) => ParamConvergence::Scored {
                        rhat: stat(&self.rhat, &name),
                        rhat_bulk: stat(&self.rhat_bulk, &name),
                        rhat_folded: stat(&self.rhat_folded, &name),
                        rhat_classic: stat(&self.rhat_classic, &name),
                        ess_bulk: stat(&self.ess, &name),
                        ess_tail: stat(&self.ess_tail, &name),
                        ess_per_chain: self
                            .ess_per_chain
                            .get(&name)
                            .cloned()
                            .unwrap_or_default(),
                        all_chains_frozen: matches!(reason, Some(RhatRefusal::NonFiniteRhat)),
                        rhat_withheld: None,
                    },
                };
                (name, entry)
            })
            .collect()
    }
}

/// Test-only: build the per-parameter map from plain `(R̂, bulk-ESS, tail-ESS)`
/// maps, so fixtures stay readable. Deliberately NOT public API — the parallel
/// maps are the shape this type exists to remove.
#[cfg(test)]
pub fn per_param_from_maps(
    rhat: BTreeMap<String, f64>,
    ess: BTreeMap<String, f64>,
    ess_tail: BTreeMap<String, f64>,
) -> BTreeMap<String, ParamConvergence> {
    ConvergenceMaps { rhat, ess, ess_tail, ..Default::default() }.per_param()
}

/// Extract a `{ "<param>": <T> }` object from a summary value. An entry that
/// does not deserialize is dropped rather than failing the whole load: a fit
/// written by a newer camdl may carry a variant this one does not know, and
/// losing one parameter's classification is better than losing the fit.
fn read_typed_map<T: serde::de::DeserializeOwned>(
    summary: &serde_json::Value,
    key: &str,
) -> BTreeMap<String, T> {
    summary
        .get(key)
        .and_then(|v| v.as_object())
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| {
                    serde_json::from_value::<T>(v.clone()).ok().map(|r| (k.clone(), r))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Like [`read_f64_map`], but a non-number entry (JSON `null`, a NaN) is kept
/// as NaN rather than dropped: for R̂, an entry that is there but has no value
/// is an unavailable statistic, and dropping it lets the parameter vanish from
/// the max-R̂ verdict.
fn read_f64_map_keeping_nan(summary: &serde_json::Value, key: &str) -> BTreeMap<String, f64> {
    summary
        .get(key)
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(f64::NAN))).collect())
        .unwrap_or_default()
}

/// The keys of a `{ "<param>": <T> }` object whose value does not deserialize
/// as `T` — what [`read_typed_map`] drops.
fn unreadable_keys<T: serde::de::DeserializeOwned>(
    summary: &serde_json::Value,
    key: &str,
) -> std::collections::BTreeSet<String> {
    summary
        .get(key)
        .and_then(|v| v.as_object())
        .map(|o| {
            o.iter()
                .filter(|(_, v)| serde_json::from_value::<T>((*v).clone()).is_err())
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Extract a `{ "<param>": f64 }` object from a summary value into a
/// `BTreeMap`. Non-finite entries (a NaN ESS serialized as JSON `null`) are
/// dropped — `v.as_f64()` yields `None`, so an ungated diagnostic reads as
/// absent rather than poisoning the map. Shared by the PGAS/PMMH loaders'
/// ESS and secondary R̂ extraction (the headline `rhat`
/// is read with [`read_f64_map_keeping_nan`].)
fn read_f64_map(summary: &serde_json::Value, key: &str) -> BTreeMap<String, f64> {
    summary
        .get(key)
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

/// Read a stage's summary JSON (`pgas_summary.json` /
/// `pmmh_summary.json` / `mh_summary.json`) into a serde value. These files
/// are written by the runners and persist scalar diagnostics that aren't in
/// fit_state.toml.
///
/// The `schema` tag is **required** (gh#728). This is the one typed loader
/// every `MethodResult` goes through, so refusing here is what keeps a
/// `rhat` written under one definition from being reported under another.
fn read_summary_json(
    stage_dir: &Path,
    filename: &str,
) -> Result<serde_json::Value, MethodResultError> {
    let path = stage_dir.join(filename);
    let contents = std::fs::read_to_string(&path).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("{}: {}", filename, e),
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&contents).map_err(|e| MethodResultError::Io {
            stage_dir: stage_dir.to_owned(),
            message: format!("{}: parse error: {}", filename, e),
        })?;
    crate::run_meta::check_fit_summary_schema(&value, filename).map_err(|message| {
        MethodResultError::Io { stage_dir: stage_dir.to_owned(), message }
    })?;
    Ok(value)
}

/// Read `<stage>/draws.tsv` and compute (n_samples, mean, q025, q975)
/// for each param in `est_names`. `draws.tsv` is the canonical
/// posterior-sample table written by both PGAS and PMMH (post-
/// burn-in, thinned, all params).
fn posterior_summaries(
    stage_dir: &Path,
    est_names: &[String],
) -> (
    usize,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
) {
    let path = stage_dir.join("draws.tsv");
    let mut mean = BTreeMap::new();
    let mut q025 = BTreeMap::new();
    let mut q975 = BTreeMap::new();
    let contents = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return (0, mean, q025, q975),
    };
    let mut lines = contents.lines();
    let header = match lines.next() {
        Some(h) => h,
        None => return (0, mean, q025, q975),
    };
    let cols: Vec<&str> = header.split('\t').collect();
    // Build column indices for the estimated-param subset.
    let mut col_idx: Vec<(String, usize)> = Vec::new();
    for name in est_names {
        if let Some(i) = cols.iter().position(|c| c == name) {
            col_idx.push((name.clone(), i));
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); col_idx.len()];
    let mut n = 0_usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        for (k, (_, ci)) in col_idx.iter().enumerate() {
            if let Some(s) = fields.get(*ci) {
                if let Ok(v) = s.parse::<f64>() {
                    samples[k].push(v);
                }
            }
        }
        n += 1;
    }
    for (k, (name, _)) in col_idx.iter().enumerate() {
        let v = &mut samples[k];
        if v.is_empty() {
            continue;
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        mean.insert(name.clone(), m);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pick = |q: f64| -> f64 {
            let idx = ((v.len() - 1) as f64 * q).round() as usize;
            v[idx.min(v.len() - 1)]
        };
        q025.insert(name.clone(), pick(0.025));
        q975.insert(name.clone(), pick(0.975));
    }
    (n, mean, q025, q975)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};

    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_methodresult_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    /// Write a `FitStage` `runid::RunRecord` leaf at `dir` carrying the
    /// `inputs` the loaders read (`method`, `n_chains`, `algorithm`).
    fn write_stage_run(dir: &Path, method: crate::run_meta::FitAlgorithm, n_chains: usize, algorithm: serde_json::Value) {
        std::fs::create_dir_all(dir).unwrap();
        let rec = serde_json::json!({
            "format_version": 1,
            "kind": "fit_stage",
            "run_id": "deadbeef".repeat(8),
            "hash_version": 1,
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {"name": "fit",   "label": "fit",   "hash": "f00d".repeat(16), "schema_version": 1},
                {"name": "method", "label": "if2", "hash": "1fb03eee00000000000000000000000000000000000000000000000000000000", "schema_version": 1},
                {"name": "seed",  "label": "seed_1", "hash": "06cbd6b300000000000000000000000000000000000000000000000000000000", "schema_version": 1}
            ],
            "status": "completed",
            "artifacts": {},
            "inputs": {
                "method": method.as_str(),
                "backend": "chain_binomial",
                "seed": 1,
                "n_chains": n_chains,
                "algorithm": algorithm,
                "best_loglik": -100.0,
                "best_chain": 0
            },
            "provenance": {"created_at": "2026-04-27T00:00:00Z", "argv": ["camdl"]}
        });
        std::fs::write(dir.join("run.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    }

    fn synthetic_if2_state() -> FitState {
        let mut start_values = std::collections::BTreeMap::new();
        start_values.insert("R0".into(), 56.8);
        start_values.insert("sigma".into(), 0.115);
        start_values.insert("N0".into(), 1000.0); // fixed param, no Â
        let mut agreement = std::collections::BTreeMap::new();
        agreement.insert("R0".into(), 1.04);
        agreement.insert("sigma".into(), 1.01);
        FitState {
            method: "scout".into(),
            seed: 42,
            timestamp: "2026-04-27T00:00:00Z".into(),
            input_hash: None,
            camdl_version: Some("0.1.0+test".into()),
            best_loglik: -3804.9,
            initial_loglik: -7000.0,
            best_chain: ChainIndex(1),
            n_chains: 4,
            n_good_chains: Some(4),
            start_values,
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            perturb_only_at_t0_params: vec![],
            chain_logliks: vec![-3810.0, -3805.0, -3812.0, -3804.9],
            chain_eval_logliks: vec![-3810.0, -3805.0, -3812.0, -3804.9],
            chain_eval_ids: Vec::new(),
            chain_eval_ses: vec![1.0, 1.0, 1.0, 1.0],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            chain_starts_kind: Some(crate::fit::starts::ChainStartsKind::Spread),
            dt_check: None,
            pf_noise: None,
            optimizer_convergence: None,
        }
    }

    /// One IF2 fit, one verdict. The fit's only Â failure is on a
    /// `perturb_only_at_t0` (initial-state) parameter, I0; every estimate is
    /// judged, so `fit table` reports a failure and the leaf is refused as a
    /// `from_mle` source.
    #[test]
    fn if2_verdict_is_the_same_for_the_table_the_summary_and_a_starts_source() {
        let tmp = tempdir("if2_t0_only");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        let mut state = synthetic_if2_state();
        state.tail_chain_agreement =
            [("R0".to_string(), 1.001), ("sigma".to_string(), 1.002), ("I0".to_string(), 16.5)]
                .into_iter().collect();
        state.perturb_only_at_t0_params = vec!["I0".into()];
        state.save(&dir.to_string_lossy()).unwrap();

        let err = crate::fit::gating::check_source_converged(dir, "@t0", false)
            .expect_err("an initial-state Â failure fails the source check");
        assert!(err.contains("I0"), "{err}");
        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.gate_verdict, GateVerdict::FailA,
            "`fit table` agrees with the source check");

        // A soft-band Â (between A_SOFT = 1.05 and a_thresh = 1.10) does not
        // excuse a NaN Â or a failing decibans leg: both are refused as a
        // source, as `fit table` reads them (incomplete / fail_db).
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        let soft = |state: &mut FitState| {
            state.resolved_gate = Some(gate.clone());
            state.perturb_only_at_t0_params = vec![];
        };
        let mut nan = synthetic_if2_state();
        soft(&mut nan);
        nan.tail_chain_agreement =
            [("R0".to_string(), 1.07), ("sigma".to_string(), f64::NAN)].into_iter().collect();
        nan.chain_eval_logliks.clear();
        nan.chain_eval_ses.clear();
        nan.save(&dir.to_string_lossy()).unwrap();
        assert!(crate::fit::gating::check_source_converged(dir, "@soft_nan", false).is_err(),
            "soft band + NaN Â is refused");
        assert_eq!(If2StageResult::load(dir).unwrap().gate_verdict, GateVerdict::Incomplete);

        let mut spread = synthetic_if2_state();
        soft(&mut spread);
        spread.tail_chain_agreement =
            [("R0".to_string(), 1.07), ("sigma".to_string(), 1.002)].into_iter().collect();
        spread.chain_eval_logliks = vec![-3800.0, -3900.0, -3810.0, -3805.0];
        spread.save(&dir.to_string_lossy()).unwrap();
        assert!(crate::fit::gating::check_source_converged(dir, "@soft_db", false).is_err(),
            "soft band + failing decibans leg is refused");
        assert_eq!(If2StageResult::load(dir).unwrap().gate_verdict, GateVerdict::FailDb);
    }

    /// A NaN Â (within-chain variance collapsed) is unavailable, and an
    /// unavailable Â is never a pass: not with no decibans leg, and not with a
    /// passing one, because chains spread along a ridge can agree in
    /// log-likelihood while disagreeing in θ. The leaf reads `incomplete` and
    /// is refused as a starts source, naming the parameter.
    #[test]
    fn a_nan_chain_agreement_is_unavailable_and_never_a_pass() {
        let tmp = tempdir("if2_nan");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        for with_decibans in [false, true] {
            let mut state = synthetic_if2_state();
            state.tail_chain_agreement =
                [("R0".to_string(), f64::NAN), ("sigma".to_string(), 1.002)].into_iter().collect();
            if !with_decibans {
                state.chain_eval_logliks.clear();
                state.chain_eval_ses.clear();
            }
            state.save(&dir.to_string_lossy()).unwrap();
            let r = If2StageResult::load(dir).unwrap();
            assert_eq!(r.gate_verdict, GateVerdict::Incomplete,
                "NaN Â on R0 (decibans leg present: {with_decibans}) is not a pass");
            let err = crate::fit::gating::check_source_converged(dir, "@nan", false)
                .expect_err("a source with an unavailable Â is refused");
            assert!(err.contains("R0"), "names the parameter: {err}");
        }
    }

    /// θ̂ is the estimated parameters only. With no Â (a one-chain IF2 run, or
    /// fewer than two converged nl-* chains), the loaders fell back to every
    /// `start_values` entry, which for IF2 includes the fixed parameters:
    /// `fit table`'s `params` (JSON) and `fit summary`'s "θ̂ (N estimated
    /// params)" then listed N0 as estimated. The estimated set is read from
    /// the fit's `fit.meta.json` instead.
    #[test]
    fn theta_hat_excludes_fixed_parameters_without_chain_agreement() {
        let tmp = tempdir("theta_hat_fixed");
        let seg = tmp.path();
        std::fs::write(seg.join("fit.meta.json"),
            serde_json::json!({"estimated": ["R0", "sigma"]}).to_string()).unwrap();
        let dir = seg.join("if2-abcd").join("seed_1-ef01");
        std::fs::create_dir_all(&dir).unwrap();
        write_stage_run(&dir, crate::run_meta::FitAlgorithm::If2, 1,
            serde_json::json!({"method": "if2", "iterations": 50}));
        let mut state = synthetic_if2_state();
        state.tail_chain_agreement.clear(); // one chain: no Â
        state.save(&dir.to_string_lossy()).unwrap();
        let r = If2StageResult::load(&dir).unwrap();
        assert_eq!(r.theta_hat.keys().cloned().collect::<Vec<_>>(), vec!["R0", "sigma"],
            "the fixed N0 is not an estimate");
    }

    /// A one-chain IF2 run computes no Â and no decibans spread: nothing was
    /// assessed, so `fit table` must not report a pass (it read converged /
    /// `pass`, because the worst Â folds from 0.0 over an empty list). The
    /// starts-source check still accepts the leaf: there is no verdict to
    /// contradict.
    #[test]
    fn a_one_chain_if2_fit_is_not_assessed() {
        let tmp = tempdir("if2_one_chain");
        let dir = tmp.path();
        write_stage_run(dir, crate::run_meta::FitAlgorithm::If2, 1,
            serde_json::json!({"method": "if2", "iterations": 50}));
        let mut state = synthetic_if2_state();
        state.method = "if2".into();
        state.n_chains = 1;
        state.tail_chain_agreement.clear();
        state.chain_eval_logliks = vec![-3804.9];
        state.chain_eval_ses = vec![1.0];
        state.save(&dir.to_string_lossy()).unwrap();
        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.gate_verdict, GateVerdict::NotAssessed);
        assert_eq!(r.max_chain_agreement, None, "no Â, not a perfect 0.0");
        crate::fit::gating::check_source_converged(dir, "@one", false)
            .expect("a one-chain source has no verdict to contradict");
        // The optimizer's verdict on one converged chain is not assessed either.
        let one = crate::fit::state::OptimizerConvergence {
            n_converged: 1, n_maxeval: 0, n_failed: 0, max_rel_range: 0.0,
            max_abs_range: 0.0, chain_agreement_failed: false, delta_loglik: 0.0,
            decibans_failed: false,
        };
        assert_eq!(one.gate_verdict(), "not_assessed");
    }

    /// With fewer than two converged nl-* chains the chain-agreement leg is
    /// not computed: the stored range is a placeholder 0.0, which must not
    /// surface as a measured agreement.
    #[test]
    fn an_uncomputed_nlopt_chain_agreement_is_none() {
        let mut r = NloptStageResult {
            best_loglik: -10.0, best_chain: ChainIndex(0), theta_hat: BTreeMap::new(),
            algorithm: "nl-sbplx".into(), n_converged: 1, max_rel_range: 0.0, n_chains: 2,
            optimizer_convergence: Some(crate::fit::state::OptimizerConvergence {
                n_converged: 1, n_maxeval: 1, n_failed: 0, max_rel_range: 0.0,
                max_abs_range: 0.0, chain_agreement_failed: false, delta_loglik: 0.0,
                decibans_failed: false,
            }),
        };
        assert_eq!(r.chain_agreement(), None, "one converged chain: not computed");
        if let Some(v) = &mut r.optimizer_convergence { v.n_converged = 2; v.n_maxeval = 0; }
        r.n_converged = 2;
        assert_eq!(r.chain_agreement(), Some(0.0));
    }

    #[test]
    fn loads_if2_stage_result_pass_verdict() {
        let tmp = tempdir("if2_pass");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();

        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.n_chains, 4);
        assert_eq!(r.n_iter, 50);
        assert_eq!(r.best_chain, ChainIndex(1));
        assert!((r.best_loglik - (-3804.9)).abs() < 1e-9);
        // theta_hat restricted to estimated params (those with Â).
        assert_eq!(r.theta_hat.len(), 2);
        assert!(r.theta_hat.contains_key("R0"));
        assert!(r.theta_hat.contains_key("sigma"));
        assert!(!r.theta_hat.contains_key("N0"),
            "fixed param N0 must not appear in theta_hat");
        // Â passes (max=1.04 < 1.01? actually 1.04 > 1.01, so this should FAIL Â).
        // Default a_thresh is 1.01, max Â = 1.04, so a_passes = false.
        // Decibans spread is small, db_passes = true.
        // Verdict: FailA.
        assert_eq!(r.gate_verdict, GateVerdict::FailA);
        assert!((r.max_chain_agreement.unwrap() - 1.04).abs() < 1e-9);
    }

    #[test]
    fn if2_pass_verdict_when_thresholds_clear() {
        let tmp = tempdir("if2_clean");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        let mut state = synthetic_if2_state();
        state.tail_chain_agreement.insert("R0".into(), 1.005);
        state.tail_chain_agreement.insert("sigma".into(), 1.002);
        state.save(&dir.to_string_lossy()).unwrap();
        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.gate_verdict, GateVerdict::Pass);
    }

    #[test]
    fn loads_if2_ess_from_chain_evaluations_tsv() {
        let tmp = tempdir("if2_ess");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 10}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        std::fs::write(
            dir.join("chain_evaluations.tsv"),
            "# camdl 0.1.0+test\n\
             chain\tcandidate\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr\tR0\n\
             1\tfinal_iter\t-3805.1\t1.0\t850.0\t412.0\t17\t0\t56.8\n\
             1\ttail_mean\t-3810.0\t1.0\t800.0\t300.0\t12\t0\t56.7\n\
             2\tfinal_iter\t-3812.0\t1.0\t820.0\t380.0\t15\t0\t57.0\n",
        )
        .unwrap();
        let r = If2StageResult::load(dir).unwrap();
        let ess = r.ess_at_mle.expect("ESS summary must be loaded");
        // Best (highest loglik) row is chain 1 / final_iter at -3805.1.
        assert!((ess.ess_min - 412.0).abs() < 1e-9);
        assert!((ess.ess_mean - 850.0).abs() < 1e-9);
        assert_eq!(ess.ess_min_step, Some(17));
    }

    #[test]
    fn ess_at_mle_none_when_file_absent() {
        let tmp = tempdir("if2_no_ess");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 10}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        let r = If2StageResult::load(dir).unwrap();
        assert!(r.ess_at_mle.is_none());
    }

    #[test]
    fn loads_pgas_stage_result() {
        let tmp = tempdir("pgas");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::Pgas,
            2,
            serde_json::json!({"method": "pgas", "sweeps": 100}),
        );
        // pgas_summary.json, schema matched to pgas::write_summary.
        std::fs::write(
            dir.join("pgas_summary.json"),
            serde_json::to_string(&serde_json::json!({
                "schema": crate::run_meta::FIT_SUMMARY_SCHEMA,
                "method": "pgas",
                "n_chains": 2,
                "acceptance_rates": [[0.32, 0.35], [0.28, 0.30]],
                "rhat": {"R0": 1.02, "sigma": 1.04},
                "ess": {"R0": 850.0, "sigma": 412.0}
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("draws.tsv"),
            "R0\tsigma\tN0\n\
             56.8\t0.115\t1000.0\n\
             57.1\t0.110\t1000.0\n\
             56.5\t0.118\t1000.0\n\
             57.3\t0.112\t1000.0\n",
        )
        .unwrap();

        let r = PgasStageResult::load(dir).unwrap();
        assert_eq!(r.diagnostics.n_chains, 2);
        assert_eq!(r.diagnostics.n_samples, 4);
        // BTreeMap order: alphabetic. R0 first, sigma second.
        let r0 = r.posterior_mean["R0"];
        assert!((r0 - (56.8 + 57.1 + 56.5 + 57.3) / 4.0).abs() < 1e-9);
        // R̂ map present, max captured.
        assert!((r.diagnostics.max_rhat().unwrap() - 1.04).abs() < 1e-9);
        // Acceptance per param: chain-mean. R0 col 0: (0.32 + 0.28)/2 = 0.30.
        assert!((r.acceptance_per_param["R0"] - 0.30).abs() < 1e-9);
        // ESS comes through.
        assert!((r.diagnostics.ess_per_param()["sigma"] - 412.0).abs() < 1e-9);
    }

    /// A `fit_state.toml` with only the fields the point-start check reads.
    fn write_state_with_starts(dir: &Path, n_chains: usize, kind: Option<&str>, source: &str) {
        let mut body = format!(
            "method = \"pgas\"\nseed = 1\ntimestamp = \"2026-01-01T00:00:00Z\"\n\
             best_loglik = -10.0\ninitial_loglik = -20.0\nbest_chain = 1\nn_chains = {n_chains}\n\
             chain_init_source = \"{source}\"\n"
        );
        if let Some(k) = kind {
            body.push_str(&format!("chain_starts_kind = \"{k}\"\n"));
        }
        body.push_str("\n[start_values]\n\n[rw_sd]\n");
        std::fs::write(dir.join("fit_state.toml"), body).unwrap();
    }

    /// Proposal 2026-09-08 §3.4: a multi-chain sampler leaf whose chains all
    /// began at one point is read with R̂ not assessed for every parameter,
    /// carrying the rule that put them there. The stored numbers are not a
    /// verdict, and the demotion is a fact about the starts, so the same
    /// summary under a spread start reports the R̂ it stored.
    #[test]
    fn a_point_started_sampler_leaf_reads_as_not_assessed() {
        let tmp = tempdir("point_start");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::Pgas,
            4,
            serde_json::json!({"method": "pgas", "sweeps": 100}),
        );
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"schema":"camdl.fit-summary/v2","method":"pgas","n_chains":4,"acceptance_rates":[],
                "rhat":{"R0":1.001,"sigma":1.002},"ess":{"R0":900.0,"sigma":880.0}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("draws.tsv"), "R0\tsigma\n56.8\t0.115\n57.1\t0.110\n").unwrap();

        // Every chain at @mle's estimate: the R̂ of 1.001 is agreement by
        // construction and must not be reported.
        write_state_with_starts(dir, 4, Some("point"), "from_mle @mle");
        let r = PgasStageResult::load(dir).unwrap();
        match r.diagnostics.max_rhat_status() {
            MaxRhat::NotApplicable { reason } => assert_eq!(reason, RhatRefusal::PointStart),
            other => panic!("a point start must read as not assessed, got {other:?}"),
        }
        assert!(!r.diagnostics.converged_at(RHAT_CONVERGED_THRESHOLD),
            "not assessed is never converged");
        for (name, p) in &r.diagnostics.per_param {
            let why = p.why_no_rhat().unwrap_or_else(|| panic!("{name} must carry the reason"));
            assert_eq!(why, "all 4 chains started at one point (starts = from_mle @mle)");
            assert!(!p.is_pathology(), "a point start is not a sampler failure");
            assert_eq!(p.rhat(), Stat::Undefined, "{name}: the R̂ is withheld");
        }
        assert_eq!(r.diagnostics.rhat_cell("R0", "—"), "—");
        // The ESS is a within-chain fact the start does not invalidate the
        // same way, so it is kept and the efficiency lines still report.
        assert!((r.diagnostics.ess_per_param()["R0"] - 900.0).abs() < 1e-9);
        assert!((r.diagnostics.min_ess().unwrap() - 880.0).abs() < 1e-9);

        // The same leaf under a spread start reports what it stored.
        write_state_with_starts(dir, 4, Some("spread"), "from_prior");
        let r = PgasStageResult::load(dir).unwrap();
        assert!((r.diagnostics.max_rhat().unwrap() - 1.002).abs() < 1e-9);
        assert!(r.diagnostics.converged_at(RHAT_CONVERGED_THRESHOLD));
    }

    /// The demotion is a claim about how the chains were started, so it fires
    /// only when the leaf makes that claim: a leaf written before
    /// `chain_starts_kind` existed, a one-chain leaf (R̂ was never applicable),
    /// and a leaf with no state at all are all left alone.
    #[test]
    fn point_start_needs_the_recorded_kind_and_two_chains() {
        let tmp = tempdir("point_start_guard");
        let dir = tmp.path();
        assert!(point_start(dir).unwrap().is_none(), "no state, no claim");
        write_state_with_starts(dir, 4, None, "from_mle @mle");
        assert!(point_start(dir).unwrap().is_none(), "a pre-split leaf makes no claim");
        write_state_with_starts(dir, 1, Some("point"), "single");
        assert!(point_start(dir).unwrap().is_none(), "one chain: R̂ was never applicable");
        write_state_with_starts(dir, 4, Some("point"), "single");
        assert_eq!(point_start(dir).unwrap(), Some((4, "single".to_string())));
        write_state_with_starts(dir, 4, Some("spread"), "lhs");
        assert!(point_start(dir).unwrap().is_none(), "a spread start is assessed");
        std::fs::write(dir.join("fit_state.toml"), "n_chains = [oops").unwrap();
        assert!(point_start(dir).is_err(),
            "a fit_state.toml that cannot be read is an error, not \"no claim\"");
    }

    #[test]
    fn loads_pmmh_stage_result() {
        let tmp = tempdir("pmmh");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::Pmmh,
            2,
            serde_json::json!({"method": "pmmh", "iterations": 50}),
        );
        std::fs::write(
            dir.join("pmmh_summary.json"),
            serde_json::to_string(&serde_json::json!({
                "schema": crate::run_meta::FIT_SUMMARY_SCHEMA,
                "method": "pmmh",
                "n_chains": 2,
                "acceptance_rate": [0.20, 0.30],
                "rhat": {"R0": 1.03},
                "ess": {"R0": 600.0},
                "map_loglik": -3801.4,
                "map_chain": 1
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("draws.tsv"),
            "R0\tN0\n\
             57.0\t1000.0\n\
             57.5\t1000.0\n",
        )
        .unwrap();

        let r = PmmhStageResult::load(dir, crate::run_meta::FitAlgorithm::Pmmh).unwrap();
        assert_eq!(r.diagnostics.n_chains, 2);
        assert_eq!(r.diagnostics.n_samples, 2);
        // Mean over the two posterior samples for R0.
        assert!((r.posterior_mean["R0"] - 57.25).abs() < 1e-9);
        assert!((r.acceptance_rate - 0.25).abs() < 1e-9);
        assert!((r.map_loglik - (-3801.4)).abs() < 1e-9);
        assert!((r.diagnostics.max_rhat().unwrap() - 1.03).abs() < 1e-9);
    }

    /// gh#590: `has_result` and `load_from` must agree on every algorithm —
    /// `true` exactly where `load_from` has an arm (so an empty leaf fails
    /// with a load error, not `UnknownMethod`), `false` exactly where it
    /// answers `UnknownMethod`.
    #[test]
    fn has_result_agrees_with_load_from() {
        use crate::run_meta::FitAlgorithm as A;
        let tmp = tempdir("has_result");
        for algo in [A::If2, A::Pgas, A::Pmmh, A::Mh, A::Pfilter,
                     A::NlSbplx, A::NlBobyqa, A::NlLbfgs] {
            let unknown = matches!(
                MethodResult::load_from(tmp.path(), algo.as_str()),
                Err(MethodResultError::UnknownMethod { .. })
            );
            assert_eq!(MethodResult::has_result(algo), !unknown,
                "has_result({}) disagrees with load_from", algo.as_str());
        }
    }

    #[test]
    fn load_from_dispatches_on_method_string() {
        let tmp = tempdir("dispatch");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 5}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        let r = MethodResult::load_from(dir, "if2").unwrap();
        assert!(matches!(r, MethodResult::If2(_)));

        let err = MethodResult::load_from(dir, "if4").unwrap_err();
        match err {
            MethodResultError::UnknownMethod { method, .. } => assert_eq!(method, "if4"),
            other => panic!("expected UnknownMethod, got {:?}", other),
        }
    }

    fn diag(ess: &[(&str, f64)], n_samples: usize, thin: usize, wall: Option<f64>) -> PosteriorDiagnostics {
        PosteriorDiagnostics {
            per_param: crate::fit::method_result::per_param_from_maps(
                [("R0", 1.02), ("sigma", 1.04)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
                ess.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                BTreeMap::new(),
            ),
            n_samples,
            thin,
            wall_time_secs: wall,
            n_chains: 2,
        }
    }

    #[test]
    fn diagnostics_max_rhat_and_min_ess_off_the_slowest_param() {
        let d = diag(&[("R0", 850.0), ("sigma", 412.0)], 500, 1, Some(11.8));
        assert!((d.max_rhat().unwrap() - 1.04).abs() < 1e-12,
            "max R̂ is the larger of the two");
        assert!((d.min_ess().unwrap() - 412.0).abs() < 1e-12, "min ESS is the slower param");
    }

    /// ESS/iteration is the algorithm-quality metric — it MUST be invariant to
    /// thinning: (500 draws, thin 1) and (50 draws, thin 10) are the same 500
    /// raw sampling iterations, so the same slowest-param ESS yields the same
    /// ESS/iter. This is the exact confound the metric was introduced to kill.
    #[test]
    fn ess_per_iter_is_thinning_invariant() {
        let unthinned = diag(&[("R0", 850.0), ("sigma", 145.0)], 500, 1, None);
        let thinned = diag(&[("R0", 850.0), ("sigma", 145.0)], 50, 10, None);
        let a = unthinned.ess_per_iter().unwrap();
        let b = thinned.ess_per_iter().unwrap();
        assert!((a - b).abs() < 1e-12, "ESS/iter invariant under thinning: {a} vs {b}");
        assert!((a - 145.0 / 500.0).abs() < 1e-12, "ESS/iter = min-param ESS / raw iters");
    }

    #[test]
    fn ess_per_sec_needs_positive_wall_time() {
        let d = diag(&[("R0", 850.0), ("sigma", 145.0)], 500, 1, Some(11.8));
        assert!((d.ess_per_sec().unwrap() - 145.0 / 11.8).abs() < 1e-9);
        // Absent or zero wall-time → None (older runs, or a zero-duration stub).
        // Both maps stay COMPLETE (every param `diag` gives an R̂ also gets an
        // ESS) so wall-time is the only thing that can produce the `None` — an
        // incomplete map would withhold the ratio for the other reason (gh#687)
        // and the assertion would stop testing what it claims.
        assert!(diag(&[("R0", 850.0), ("sigma", 145.0)], 500, 1, None).ess_per_sec().is_none());
        assert!(diag(&[("R0", 850.0), ("sigma", 145.0)], 500, 1, Some(0.0)).ess_per_sec().is_none());
    }

    /// Build diagnostics with an explicit R̂ map, so a test can express "this
    /// parameter was assessed across chains (finite R̂) but reports no pooled
    /// ESS" — the state gh#687 is about.
    fn diag_rhat(
        rhat: &[(&str, f64)],
        ess: &[(&str, f64)],
        n_samples: usize,
        wall: Option<f64>,
    ) -> PosteriorDiagnostics {
        PosteriorDiagnostics {
            per_param: crate::fit::method_result::per_param_from_maps(
                rhat.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                ess.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                BTreeMap::new(),
            ),
            n_samples,
            thin: 1,
            wall_time_secs: wall,
            n_chains: 4,
        }
    }

    /// gh#687: a parameter can reach the map with no bulk ESS at all — a NaN
    /// serializes to JSON `null` and `read_f64_map` drops it, so the key is
    /// ABSENT. The minimum must not be taken over the parameters that survived
    /// — that is a minimum over a subset, reported as if it bounded the whole
    /// posterior, and it RISES as the fit gets worse.
    #[test]
    fn min_ess_is_undefined_when_an_assessed_param_reports_no_ess() {
        // rho was assessed (R̂ = 2.639) and reports no ESS; the other two did.
        let d = diag_rhat(
            &[("I0", 1.02), ("k_cases", 1.03), ("rho", 2.639)],
            &[("I0", 3913.0), ("k_cases", 559.0)],
            43_000,
            Some(3600.0),
        );
        assert!(
            d.min_ess().is_none(),
            "min ESS must be undefined while rho has none, not 559 over the survivors: {:?}",
            d.min_ess()
        );
        assert!(d.ess_per_iter().is_none(), "ESS/iter inherits the undefined minimum");
        assert!(d.ess_per_sec().is_none(), "ESS/sec inherits the undefined minimum");
        match d.min_ess_status() {
            MinEss::Unreportable { missing, n_expected } => {
                assert_eq!(missing, vec!["rho".to_string()], "the diagnosis names the parameter");
                assert_eq!(n_expected, 3, "3 parameters were assessed across chains");
            }
            other => panic!("expected Unreportable naming rho, got {other:?}"),
        }
    }

    /// The `--exclude-chains` recompute (`chain_selection::recompute_subset_
    /// diagnostics`) keeps the key and stores the suppressed ESS as NaN rather
    /// than dropping it. `f64::min` returns the non-NaN operand, so the NaN is
    /// walked past silently — the same inversion by a different encoding.
    #[test]
    fn min_ess_is_undefined_when_an_ess_entry_is_nan() {
        let d = diag_rhat(
            &[("I0", 1.02), ("rho", 2.639)],
            &[("I0", 3913.0), ("rho", f64::NAN)],
            43_000,
            Some(3600.0),
        );
        assert!(
            d.min_ess().is_none(),
            "a NaN ESS entry must not be skipped into a subset minimum: {:?}",
            d.min_ess()
        );
        match d.min_ess_status() {
            MinEss::Unreportable { missing, .. } => assert_eq!(missing, vec!["rho".to_string()]),
            other => panic!("expected Unreportable naming rho, got {other:?}"),
        }
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

    /// The per-chain ESS must survive the round trip through
    /// `*_summary.json` to the surface. It was written by every sampler and
    /// read by nothing, so the follow-up question to a high R̂ — is each chain
    /// mixing inside its own mode, or is one stuck — had no answer anywhere a
    /// user looks.
    #[test]
    fn the_per_chain_ess_reaches_the_surface_from_a_stored_summary() {
        let summary = serde_json::json!({
            "rhat": { "beta": 2.4 },
            "ess": { "beta": 12.0 },
            "ess_per_chain": { "beta": [410.0, 388.0, 9.0] },
        });
        let per_param = ConvergenceMaps::read(&summary).per_param();
        assert_eq!(
            per_param["beta"].ess_per_chain(),
            &[410.0, 388.0, 9.0],
            "every chain's value survives, in chain order — the SPREAD is the \
             diagnosis, so one dropped cell changes the reading"
        );

        // A parameter the estimator refused carries no per-chain values at
        // all, which is a different statement from "all its chains mixed".
        let refused = serde_json::json!({
            "rhat_not_reported": { "beta": "too_few_chains" },
        });
        assert!(
            ConvergenceMaps::read(&refused).per_param()["beta"].ess_per_chain().is_empty(),
            "a refusal carries no per-chain ESS, which is not the same fact as \
             a parameter whose chains all mixed"
        );
    }

    /// A parameter must never vanish from the max-R̂ verdict. A stored R̂ of
    /// `null` (a NaN, as serde_json writes one) and a `rhat_not_reported`
    /// entry this build cannot read both used to drop the parameter, so the
    /// max over the rest read as converged. Either is an unavailable R̂, and
    /// the verdict is then not a pass.
    #[test]
    fn a_missing_or_unreadable_rhat_is_never_a_pass() {
        let verdict = |summary: serde_json::Value| PosteriorDiagnostics {
            per_param: ConvergenceMaps::read(&summary).per_param(),
            n_samples: 100, thin: 1, wall_time_secs: None, n_chains: 4,
        }.max_rhat_status();
        let null_rhat = serde_json::json!({
            "rhat": { "beta": 1.001, "gamma": null },
            "ess": { "beta": 900.0, "gamma": 800.0 },
        });
        match verdict(null_rhat) {
            MaxRhat::Unassessable { params } => assert_eq!(params, vec!["gamma".to_string()]),
            other => panic!("a null R̂ on gamma must not read as converged: {other:?}"),
        }
        let unknown_refusal = serde_json::json!({
            "rhat": { "beta": 1.001 },
            "ess": { "beta": 900.0 },
            "rhat_not_reported": { "gamma": "a_reason_from_a_newer_camdl" },
        });
        match verdict(unknown_refusal) {
            MaxRhat::Unassessable { params } => assert_eq!(params, vec!["gamma".to_string()]),
            other => panic!("gamma's unreadable refusal must not drop it: {other:?}"),
        }
        // A recorded structural refusal is still "not assessed", not a failure.
        let too_few = serde_json::json!({ "rhat_not_reported": { "beta": "too_few_chains" } });
        assert!(matches!(verdict(too_few), MaxRhat::NotApplicable { .. }));
    }

    /// A parameter whose R̂ is `∞` — every chain frozen at its own value — is a
    /// SAMPLER PATHOLOGY. It must sink the verdict even when other parameters
    /// reported perfectly well: a fit is not converged because some of it was.
    #[test]
    fn an_infinite_rhat_sinks_the_verdict_even_beside_healthy_params() {
        let d = PosteriorDiagnostics {
            per_param: BTreeMap::from([
                ("beta".to_string(), ParamConvergence::Scored {
                    rhat: Stat::Value(1.01),
                    rhat_bulk: Stat::Value(1.01),
                    rhat_folded: Stat::Value(1.00),
                    rhat_classic: Stat::Value(1.00),
                    ess_bulk: Stat::Value(900.0),
                    ess_tail: Stat::Value(850.0),
                    ess_per_chain: Vec::new(),
                    all_chains_frozen: false,
                    rhat_withheld: None,
                }),
                ("frozen".to_string(), ParamConvergence::Scored {
                    rhat: Stat::Infinite,
                    rhat_bulk: Stat::Infinite,
                    rhat_folded: Stat::Undefined,
                    rhat_classic: Stat::Infinite,
                    ess_bulk: Stat::Value(3.0),
                    ess_tail: Stat::Undefined,
                    ess_per_chain: Vec::new(),
                    all_chains_frozen: true,
                    rhat_withheld: None,
                }),
            ]),
            n_samples: 4000,
            thin: 1,
            wall_time_secs: Some(60.0),
            n_chains: 4,
        };

        match d.max_rhat_status() {
            MaxRhat::Unassessable { params } => assert_eq!(params, vec!["frozen".to_string()]),
            other => panic!("a frozen parameter must make the headline Unassessable, got {other:?}"),
        }
        assert!(!d.converged_at(RHAT_CONVERGED_THRESHOLD),
            "a fit containing a frozen parameter is NOT converged");
        assert_eq!(d.max_rhat(), None, "and publishes no headline number");
        // The healthy parameter is untouched and still readable.
        assert_eq!(d.rhat_cell("beta", "—"), "1.010");
        assert_eq!(d.rhat_cell("frozen", "—"), "∞", "∞ is shown, not hidden");
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

    /// A structural refusal is "not assessed", which is NOT "converged" — the
    /// distinction option C exists for.
    #[test]
    fn a_structural_refusal_is_not_assessed_and_not_converged() {
        let d = PosteriorDiagnostics {
            per_param: BTreeMap::from([
                ("beta".to_string(),
                 ParamConvergence::NotScored {
                     reason: RhatRefusal::TooFewChains, detail: None }),
            ]),
            n_samples: 100, thin: 1, wall_time_secs: None, n_chains: 1,
        };
        match d.max_rhat_status() {
            MaxRhat::NotApplicable { reason } => assert_eq!(reason, RhatRefusal::TooFewChains),
            other => panic!("expected NotApplicable, got {other:?}"),
        }
        assert!(!d.converged_at(RHAT_CONVERGED_THRESHOLD),
            "a single-chain fit was never assessed, so it is not converged");
    }

    /// A parameter the model PINS never enters the map at all, so it cannot
    /// withhold the headline — and that is now true by construction rather
    /// than by a filter.
    ///
    /// This test used to assert the opposite shape: that constant columns
    /// swept in by `fit predict`'s all-columns subset recompute carried a
    /// non-finite R̂ and had to be skipped. That recompute is gone. The subset
    /// path iterates the ESTIMATED set read from the fit's own sidecar, so the
    /// only way a NaN reaches this map is an estimated parameter that failed —
    /// which MUST withhold. One filter was doing two jobs; separating them is
    /// what made a frozen parameter distinguishable from a pinned one.
    #[test]
    fn a_pinned_param_is_absent_from_the_map_not_filtered_out_of_it() {
        // Exactly the estimated set. `k` and `rho` are pinned by the model and
        // are simply not here.
        let d = diag_rhat(
            &[("beta", 1.01), ("gamma", 1.02)],
            &[("beta", 300.0), ("gamma", 145.0)],
            500,
            Some(11.8),
        );
        assert_eq!(
            d.min_ess_status(),
            MinEss::Reported(145.0),
            "every estimated parameter reported, so the headline is the slowest"
        );
        assert!((d.ess_per_iter().unwrap() - 145.0 / 500.0).abs() < 1e-12);
        assert!(!d.per_param.contains_key("k") && !d.per_param.contains_key("rho"),
            "a pinned parameter is not in the estimated set: {:?}", d.per_param.keys());

        // Control: the same map with one ESTIMATED parameter failing DOES
        // withhold. A NaN here can only mean a sampler failure now.
        let broken = diag_rhat(
            &[("beta", 1.01), ("gamma", 1.02)],
            &[("beta", 300.0), ("gamma", f64::NAN)],
            500,
            Some(11.8),
        );
        match broken.min_ess_status() {
            MinEss::Unreportable { missing, .. } => assert_eq!(missing, vec!["gamma".to_string()]),
            other => panic!("an estimated parameter with no ESS must withhold, got {other:?}"),
        }
    }

    /// THE property this issue exists for. A strictly better fit — every
    /// parameter that reported before still reports, at least as well, and MORE
    /// parameters report — must never carry a WORSE efficiency headline.
    ///
    /// Measured inversion (gh#687), two runs of one model differing only in
    /// particle count: N=1200 reported max R̂ 2.639 / min-param ESS 559 /
    /// ESS/iter 0.013; N=4800 reported max R̂ 1.455 / min-param ESS 73 /
    /// ESS/iter 0.001 — 13x worse for the better fit, purely because it
    /// converged more parameters into the map.
    #[test]
    fn efficiency_never_inverts_when_more_params_converge() {
        fn assert_no_inversion(worse: &PosteriorDiagnostics, better: &PosteriorDiagnostics, case: &str) {
            match (worse.ess_per_iter(), better.ess_per_iter()) {
                (Some(w), Some(b)) => assert!(
                    w <= b,
                    "{case}: the worse fit reports ESS/iter {w} against the better fit's {b}"
                ),
                (Some(w), None) => panic!(
                    "{case}: the worse fit reports ESS/iter {w} while the better reports nothing"
                ),
                (None, _) => {}
            }
        }

        // (i) The gh#687 pair: `rho` mixes badly at N=1200 and is absent from
        //     the map; at N=4800 it converges and reports 73.
        let n1200 = diag_rhat(
            &[("I0", 1.02), ("k_cases", 1.03), ("rho", 2.639)],
            &[("I0", 3913.0), ("k_cases", 559.0)],
            43_000,
            Some(3600.0),
        );
        let n4800 = diag_rhat(
            &[("I0", 1.02), ("k_cases", 1.03), ("rho", 1.05)],
            &[("I0", 3913.0), ("k_cases", 559.0), ("rho", 73.0)],
            43_000,
            Some(3600.0),
        );
        assert!(
            n4800.ess_per_iter().is_some(),
            "the fully-reporting fit must still get a number — withholding both \
             would satisfy the ordering vacuously"
        );
        assert_no_inversion(&n1200, &n4800, "more params converged");

        // (ii) Both fully reporting, the second better on the binding parameter
        //      — the ordering is compared as two numbers, not short-circuited
        //      by an undefined side.
        let slower = diag_rhat(
            &[("I0", 1.02), ("rho", 1.05)],
            &[("I0", 3913.0), ("rho", 73.0)],
            43_000,
            Some(3600.0),
        );
        let faster = diag_rhat(
            &[("I0", 1.02), ("rho", 1.03)],
            &[("I0", 3913.0), ("rho", 150.0)],
            43_000,
            Some(3600.0),
        );
        assert!(slower.ess_per_iter().is_some() && faster.ess_per_iter().is_some());
        assert_no_inversion(&slower, &faster, "same params, better mixing");
    }

    #[test]
    fn empty_diagnostics_yield_none_not_nan() {
        let d = PosteriorDiagnostics {
            per_param: crate::fit::method_result::per_param_from_maps(
                BTreeMap::new(),
                BTreeMap::new(),
                BTreeMap::new(),
            ),
            n_samples: 0,
            thin: 1,
            wall_time_secs: Some(5.0),
            n_chains: 1,
        };
        // Review blocker 1: an empty map is NOT `0.0`. Folding from zero made a
        // fit whose R̂ could not be computed report `max R̂ = 0.000 ✓` and
        // `converged: true` — the assertion below used to pin that.
        assert_eq!(d.max_rhat_status(), MaxRhat::NoParams,
            "no params → say so, never a number that reads as converged");
        assert_eq!(d.max_rhat(), None);
        assert!(!d.converged_at(RHAT_CONVERGED_THRESHOLD),
            "a fit with no R̂ at all must not be reported as converged");
        assert!(d.min_ess().is_none());
        assert!(d.ess_per_iter().is_none(), "no samples → None, no divide-by-zero");
        assert!(d.ess_per_sec().is_none());
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
}
