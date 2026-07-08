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
