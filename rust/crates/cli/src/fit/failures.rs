//! What a predictive run computed, and what refused.
//!
//! A _deterministic failure_ is a fact about the model as written that holds
//! for whatever produced it regardless of how often it occurs: a horizon that
//! cannot be resolved, a scenario window this verb cannot honour, a `NaN`/±∞
//! where a number was about to be banded. One occurrence is a defect, so the
//! run names the site — and the draw, where there is one — and exits 1.
//!
//! The rule this module exists to enforce is that failing closed on the exit
//! status is not the same as discarding a complete object. `fit predict`
//! computes many things that fail independently — the data-conditioned
//! one-step-ahead band, which never reaches the forecast horizon; the
//! free-forward tail; one `quantities {}` entry among twenty — and refusing to
//! write the others because one refused loses work that was finished and
//! correct. So: compute what is computable, write it, name what failed in
//! `report.json`, and pass nothing
//! (proposal `docs/dev/proposals/2026-09-08-workflow-first-fit-config.md` §3.5
//! and §8 item 15).

use std::path::{Path, PathBuf};

use serde::Serialize;

/// The schema tag of `report.json`. Bumped by any change to the failure shape,
/// since it is the only thing telling a consumer what contract the file was
/// written under.
pub const REPORT_SCHEMA: &str = "camdl.predict-report/v1";

/// Where a deterministic failure occurred.
///
/// Serializes externally-tagged: `"free_forward"` for the unit variant,
/// `{"quantity": "<name>"}` for the named one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    /// The free-forward tail as a whole — the replay past the last observation.
    /// Not attributable to one stream or one quantity: when its production
    /// refuses, no cell of it exists.
    FreeForward,
    /// One `quantities {}` entry, by its declared name. The other entries and
    /// every stream artifact are unaffected by it and are still written.
    Quantity(String),
}

impl Site {
    /// The phrase this site takes in a sentence on stderr.
    pub fn describe(&self) -> String {
        match self {
            Site::FreeForward => "the free-forward tail".to_string(),
            Site::Quantity(name) => format!("quantity `{name}`"),
        }
    }
}

/// One deterministic failure, internally tagged under `kind` so a consumer
/// reads `{"kind": "evaluation_failed", "at": "free_forward", "reason": …}`.
///
/// `EvaluationFailed` is the whole-site kind — a horizon or a replay refused
/// and no single draw is to blame. `NonFinite` names the draw whose value could
/// not be published.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeterministicFailure {
    /// A step of the run refused: an unresolvable horizon, a scenario window
    /// this verb cannot honour, a rate or integrator that errored.
    EvaluationFailed { at: Site, reason: String },
    /// A `NaN` or ±∞ reached a value that was about to be banded, on `draw` —
    /// an index into the draws this run replayed, not into the stored cloud.
    NonFinite { at: Site, draw: usize, reason: String },
}

impl DeterministicFailure {
    /// The one line this failure prints on stderr.
    pub fn describe(&self) -> String {
        match self {
            DeterministicFailure::EvaluationFailed { at, reason } => {
                format!("{}: {reason}", at.describe())
            }
            DeterministicFailure::NonFinite { at, draw, reason } => {
                format!("{}, draw {draw}: {reason}", at.describe())
            }
        }
    }
}

/// Write `report.json` beside the predictive artifacts and return its path.
///
/// Written on every run, not only a failing one: `"failures": []` is the
/// statement that the run computed everything it set out to, and a consumer
/// that has to distinguish "no failures" from "no report" by the file's
/// absence cannot tell either from a run that crashed before writing.
pub fn write_report(
    segment: &Path,
    name: &str,
    failures: &[DeterministicFailure],
) -> Result<PathBuf, String> {
    let doc = serde_json::json!({
        "schema": REPORT_SCHEMA,
        "failures": failures,
    });
    let path = segment.join(format!("{name}.json"));
    let text = serde_json::to_string_pretty(&doc)
        .map_err(|e| format!("serializing the predictive report: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_run_writes_an_empty_failure_list() {
        let dir = crate::test_support::unique_temp_dir("predict_report_clean");
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_report(&dir, "report", &[]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["schema"], REPORT_SCHEMA);
        assert_eq!(
            v["failures"],
            serde_json::json!([]),
            "an empty list is the statement that nothing refused, not an absent key"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_failure_names_its_site_its_draw_and_its_reason_in_the_json() {
        let dir = crate::test_support::unique_temp_dir("predict_report_failed");
        std::fs::create_dir_all(&dir).unwrap();
        let failures = vec![
            DeterministicFailure::EvaluationFailed {
                at: Site::FreeForward,
                reason: "scenario 'longer' declares a horizon of t = 160".to_string(),
            },
            DeterministicFailure::NonFinite {
                at: Site::Quantity("growth".to_string()),
                draw: 37,
                reason: "3 of 200 draws non-finite".to_string(),
            },
        ];
        let path = write_report(&dir, "report", &failures).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["failures"][0]["kind"], "evaluation_failed");
        assert_eq!(v["failures"][0]["at"], "free_forward");
        assert!(v["failures"][0]["reason"].as_str().unwrap().contains("t = 160"));
        assert_eq!(v["failures"][1]["kind"], "non_finite");
        assert_eq!(
            v["failures"][1]["at"]["quantity"], "growth",
            "a quantity failure names the entry, so a reader knows which of \
             twenty dropped out"
        );
        assert_eq!(v["failures"][1]["draw"], 37);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn describe_reads_as_a_sentence() {
        assert_eq!(
            DeterministicFailure::EvaluationFailed {
                at: Site::FreeForward,
                reason: "no output time past the data".to_string(),
            }
            .describe(),
            "the free-forward tail: no output time past the data"
        );
        assert_eq!(
            DeterministicFailure::NonFinite {
                at: Site::Quantity("growth".to_string()),
                draw: 3,
                reason: "division by an empty compartment".to_string(),
            }
            .describe(),
            "quantity `growth`, draw 3: division by an empty compartment"
        );
    }
}
