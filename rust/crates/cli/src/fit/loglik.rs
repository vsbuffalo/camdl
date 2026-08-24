//! The *class* of a reported log-likelihood, carried onto every surface
//! that displays or serializes one (gh#280).
//!
//! camdl reports two classes of log-likelihood, which are not
//! interchangeable:
//!
//! - **marginal** `log p(y | θ)` — a function of θ alone, the quantity
//!   model/chain comparison needs. Produced by IF2 (a clean high-particle
//!   PF re-eval at θ̂), PMMH (`map_loglik`), and the NLopt / MH ODE
//!   skeleton fits.
//! - **complete-data (joint)** `log p(y, x | θ) = transition_ll + obs_ll`
//!   — a function of the *sampled trajectory* `x`, not θ alone. Produced
//!   by PGAS, whose Gibbs target is `(θ, x)`. It is **not** comparable to a
//!   marginal and is gameable: a chain can raise it by finding a smoother
//!   trajectory (`transition_ll ↑`) at the cost of data fit (`obs_ll ↓`).
//!
//! The kind is a property of the inference *method*, so it is derived once
//! — via [`From<FitAlgorithm>`] and [`From<&MethodResult>`] — and read by
//! `FitState`, the progress feed, the artifact writers, and the human
//! headlines. No consumer re-spells it as a free string; the producer
//! sites (`FitState`) declare the same kind by construction, cross-checked
//! in the tests below.

use serde::{Deserialize, Serialize};

use crate::fit::method_result::MethodResult;
use crate::run_meta::FitAlgorithm;

/// The marginal samplers' (`pmmh` / `mh` / `nuts`) per-iteration
/// `log p(y | θ)` column in `chain_N/trace.tsv`.
pub const TRACE_COL_LOG_LIKELIHOOD: &str = "log_likelihood";

/// PGAS's per-sweep `log p(y, X | θ)` column — its Gibbs target, evaluated
/// at the sweep's conditioned path `X`.
pub const TRACE_COL_COMPLETE_DATA_LL: &str = "log_complete_data_ll";

/// PGAS's latent-path term `log p(X | θ)`: the process density of the
/// conditioned path. A *density at one path*, not an integral over paths.
pub const TRACE_COL_TRANSITION_LL: &str = "transition_ll";

/// PGAS's observation term `log p(y | X, θ)`: how well the sweep's
/// conditioned path reproduces the observed data.
pub const TRACE_COL_OBS_LL: &str = "obs_ll";

/// The `obs_ll` column resolved by observation stream: `obs_ll_<stream>` for
/// each stream the model declares, so a fit can be asked WHICH stream it is
/// straining against without re-running the filter (gh#742).
///
/// One column per stream, not per stream × stratum: an indexed stream's column
/// sums over its strata. Per-stratum scores answer "which district", a different
/// question with its own machinery, and would make a 774-unit model's trace
/// thousands of columns wide.
///
/// Every column is populated on every row — a sweep evaluates the whole
/// likelihood, so a stream on its own cadence still contributes its own sum.
/// The columns add up to [`TRACE_COL_OBS_LL`] to floating-point round-off (the
/// two sums associate in different orders).
pub fn trace_col_obs_ll_stream(stream: &str) -> String {
    format!("{TRACE_COL_OBS_LL}_{stream}")
}

/// PGAS's initial-state term `log p(x₀ | θ)`, from the laws the model declares
/// in `init { }`. Zero for a deterministic `init { }`.
///
/// Written as its own column rather than left to be recovered by subtracting
/// the other two from `log_complete_data_ll`: a constant component of the
/// target that is only visible by subtraction is what made gh#719 need trace
/// forensics to find.
pub const TRACE_COL_INITIAL_STATE_LL: &str = "initial_state_ll";

/// The comparison class of a log-likelihood value. Serializes to the same
/// `snake_case` tags the codebase used as free strings before gh#280
/// (`"if2"`, `"marginal"`, `"ode_marginal"`, `"complete_data"`), so legacy
/// `fit_state.toml` files deserialize unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoglikType {
    /// IF2 clean-eval marginal at θ̂ (iterated-filtering MLE).
    If2,
    /// Generic marginal `log p(y | θ)`: PMMH MAP, standalone `pfilter`,
    /// `survey` / `profile` grid evaluations.
    Marginal,
    /// Marginal on the deterministic ODE skeleton: NLopt MLE and MH-ODE.
    OdeMarginal,
    /// Complete-data / joint `log p(y, x | θ)` from PGAS — **not**
    /// comparable to a marginal.
    CompleteData,
}

impl LoglikType {
    /// The serialized / wire tag, identical to the `snake_case` serde form.
    pub fn tag(self) -> &'static str {
        match self {
            LoglikType::If2 => "if2",
            LoglikType::Marginal => "marginal",
            LoglikType::OdeMarginal => "ode_marginal",
            LoglikType::CompleteData => "complete_data",
        }
    }

    /// Whether this is a marginal `log p(y | θ)`. Defined by **inclusion** —
    /// the three marginal kinds — so any future non-marginal variant (e.g. an
    /// observation-conditional `log p(y | x, θ)`) defaults to non-marginal
    /// rather than silently joining the marginals.
    ///
    /// This answers "same *kind*", NOT "safe to subtract": two marginals from
    /// different process models (ODE-deterministic vs chain-binomial PF) are
    /// not on the same scale. Comparability additionally requires an equal
    /// backend, which a consumer must check separately — never read this as a
    /// subtractable signal.
    pub fn is_marginal(self) -> bool {
        matches!(
            self,
            LoglikType::If2 | LoglikType::Marginal | LoglikType::OdeMarginal
        )
    }

    /// The `chain_N/trace.tsv` column whose per-chain mean may be compared
    /// ACROSS chains for a sampler of this class — the input the per-chain
    /// outlier score is entitled to (gh#667).
    ///
    /// - The three marginal kinds stream `log p(y | θ)` in
    ///   [`TRACE_COL_LOG_LIKELIHOOD`]. It is a function of θ alone, so two
    ///   chains' values differ only through θ: directly comparable.
    /// - `CompleteData` (PGAS) streams `log p(y, X | θ)` in
    ///   [`TRACE_COL_COMPLETE_DATA_LL`], which is **not** comparable: every
    ///   chain conditions on its own sampled path `X`, and the latent-path
    ///   term `log p(X | θ)` is a density at one path, not the marginal
    ///   `∫ p(y|X) p(X|θ) dX`. A θ whose path distribution is more
    ///   concentrated raises it for every typical path with no better fit to
    ///   the data, so ranking on it ranks chains by the entropy of their path
    ///   distribution. On the 60,000-sweep fit that motivated gh#667 the
    ///   between-chain spread was 522 nats in the path term and 9 nats in the
    ///   observation term. PGAS is therefore scored on [`TRACE_COL_OBS_LL`],
    ///   `log p(y | X, θ)` — the term that answers "does this chain reproduce
    ///   the data".
    ///
    /// `obs_ll` is **not** a marginal, so this is deliberately not derived
    /// from [`is_marginal`](Self::is_marginal): it is the part of the
    /// complete-data target the data enters, which is what makes it
    /// comparable chain-to-chain. The match is exhaustive on purpose — a new
    /// variant must state its column rather than inherit one.
    pub fn chain_agreement_column(self) -> &'static str {
        match self {
            LoglikType::If2 | LoglikType::Marginal | LoglikType::OdeMarginal => {
                TRACE_COL_LOG_LIKELIHOOD
            }
            LoglikType::CompleteData => TRACE_COL_OBS_LL,
        }
    }

    /// Progress-feed metric prefix: `ll(complete)=` for PGAS's complete-data
    /// value, `ll=` for a marginal `log p(y | θ)`. "complete" echoes the
    /// `log_complete_data_ll` trace column and the `complete_data` tag — the
    /// codebase's established term — rather than the ambiguous "joint" (joint
    /// over *what*?). It is legible to a human watching the bar yet still a
    /// *distinct* key from `ll=`, so a scraper grepping `ll=` does not pick up
    /// the non-comparable complete-data value.
    pub fn metric_prefix(self) -> &'static str {
        if self.is_marginal() { "ll" } else { "ll(complete)" }
    }

    /// The tag for an optional type — `"unknown"` for a legacy / absent
    /// value (never inferred). The single rendering used by every artifact
    /// and headline so absence reads the same everywhere.
    pub fn tag_or_unknown(this: Option<LoglikType>) -> &'static str {
        this.map_or("unknown", LoglikType::tag)
    }
}

impl std::fmt::Display for LoglikType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tag())
    }
}

/// The loglik class is a function of the run's algorithm. This covers the
/// surfaces keyed on the broad `FitAlgorithm` (browse, profile, pfilter,
/// survey).
impl From<FitAlgorithm> for LoglikType {
    fn from(a: FitAlgorithm) -> Self {
        match a {
            FitAlgorithm::If2 => LoglikType::If2,
            FitAlgorithm::Pgas => LoglikType::CompleteData,
            FitAlgorithm::Pmmh => LoglikType::Marginal,
            FitAlgorithm::Pfilter => LoglikType::Marginal,
            // Deterministic ODE-skeleton marginals.
            FitAlgorithm::Mh
            | FitAlgorithm::Nuts
            | FitAlgorithm::NlSbplx
            | FitAlgorithm::NlBobyqa => LoglikType::OdeMarginal,
        }
    }
}

/// The same map keyed on the typed fit-stage result (fit-summary / fit-table
/// consumers, which hold a `MethodResult`).
impl From<&MethodResult> for LoglikType {
    fn from(r: &MethodResult) -> Self {
        match r {
            MethodResult::If2(_) => LoglikType::If2,
            MethodResult::Pgas(_) => LoglikType::CompleteData,
            MethodResult::Pmmh(_) => LoglikType::Marginal,
            // nuts samples the deterministic ODE marginal likelihood (same kind
            // as mh-on-ode / the NLopt ODE optimizer).
            MethodResult::Nuts(_) => LoglikType::OdeMarginal,
            MethodResult::Nlopt(_) => LoglikType::OdeMarginal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_matches_legacy_strings() {
        assert_eq!(LoglikType::If2.tag(), "if2");
        assert_eq!(LoglikType::Marginal.tag(), "marginal");
        assert_eq!(LoglikType::OdeMarginal.tag(), "ode_marginal");
        assert_eq!(LoglikType::CompleteData.tag(), "complete_data");
    }

    #[test]
    fn serde_form_equals_the_pre_gh280_free_strings() {
        // Pins fit_state.toml back-compat: the enum must (de)serialize to
        // the exact strings the producers wrote before gh#280.
        for (kind, s) in [
            (LoglikType::If2, "\"if2\""),
            (LoglikType::Marginal, "\"marginal\""),
            (LoglikType::OdeMarginal, "\"ode_marginal\""),
            (LoglikType::CompleteData, "\"complete_data\""),
        ] {
            assert_eq!(serde_json::to_string(&kind).unwrap(), s);
            let back: LoglikType = serde_json::from_str(s).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn is_marginal_is_by_inclusion_so_only_the_three_marginals_qualify() {
        // By inclusion (not `!= CompleteData`): a future non-marginal kind
        // must opt in, it cannot default into the marginals.
        assert!(LoglikType::If2.is_marginal());
        assert!(LoglikType::Marginal.is_marginal());
        assert!(LoglikType::OdeMarginal.is_marginal());
        assert!(!LoglikType::CompleteData.is_marginal());
    }

    /// gh#667: the per-chain comparison column is a property of the CLASS, and
    /// the complete-data class does not get to nominate its own target. PGAS
    /// scores on `obs_ll`; the marginal kinds score on `log_likelihood`.
    #[test]
    fn chain_agreement_column_never_ranks_on_the_complete_data_target() {
        assert_eq!(LoglikType::CompleteData.chain_agreement_column(), "obs_ll");
        assert_ne!(
            LoglikType::CompleteData.chain_agreement_column(),
            TRACE_COL_COMPLETE_DATA_LL,
            "log p(y, X | θ) is not comparable across chains — every chain \
             conditions on its own X (gh#667)"
        );
        for kind in [LoglikType::If2, LoglikType::Marginal, LoglikType::OdeMarginal] {
            assert_eq!(
                kind.chain_agreement_column(),
                "log_likelihood",
                "{kind}'s trace column 1 already IS log p(y | θ)"
            );
        }
    }

    /// gh#742: a per-stream column is `obs_ll`'s name with the stream appended,
    /// so the decomposition reads as a refinement of the column it sums to and a
    /// reader can find every part by the one prefix. The scored column stays
    /// `obs_ll` exactly — a per-stream column must never be mistaken for it.
    #[test]
    fn per_stream_column_extends_the_obs_ll_name_without_colliding_with_it() {
        assert_eq!(trace_col_obs_ll_stream("cases"), "obs_ll_cases");
        assert_eq!(trace_col_obs_ll_stream("community_deaths"), "obs_ll_community_deaths");
        for stream in ["cases", "deaths"] {
            let col = trace_col_obs_ll_stream(stream);
            assert!(col.starts_with(TRACE_COL_OBS_LL), "{col} must extend {TRACE_COL_OBS_LL}");
            assert_ne!(
                col,
                LoglikType::CompleteData.chain_agreement_column(),
                "a per-stream column must not shadow the column PGAS chains are \
                 ranked on (gh#667)"
            );
        }
    }

    #[test]
    fn metric_prefix_marks_complete_data_legibly() {
        assert_eq!(LoglikType::CompleteData.metric_prefix(), "ll(complete)");
        assert_eq!(LoglikType::If2.metric_prefix(), "ll");
        assert_eq!(LoglikType::Marginal.metric_prefix(), "ll");
        assert_eq!(LoglikType::OdeMarginal.metric_prefix(), "ll");
        // `ll(complete)` does not contain the bare `ll=` key, so a marginal
        // scraper never picks up the complete-data value.
        assert!(!"ll(complete)=".contains("ll="));
    }

    #[test]
    fn tag_or_unknown_never_infers() {
        assert_eq!(LoglikType::tag_or_unknown(None), "unknown");
        assert_eq!(
            LoglikType::tag_or_unknown(Some(LoglikType::CompleteData)),
            "complete_data"
        );
    }

    #[test]
    fn algorithm_mapping_is_total_and_honest() {
        use FitAlgorithm::*;
        assert_eq!(LoglikType::from(If2), LoglikType::If2);
        assert_eq!(LoglikType::from(Pgas), LoglikType::CompleteData);
        assert_eq!(LoglikType::from(Pmmh), LoglikType::Marginal);
        assert_eq!(LoglikType::from(Pfilter), LoglikType::Marginal);
        assert_eq!(LoglikType::from(Mh), LoglikType::OdeMarginal);
        assert_eq!(LoglikType::from(NlSbplx), LoglikType::OdeMarginal);
        assert_eq!(LoglikType::from(NlBobyqa), LoglikType::OdeMarginal);
    }
}
