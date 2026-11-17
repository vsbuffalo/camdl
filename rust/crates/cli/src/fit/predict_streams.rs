//! Which of the model's observation streams a `fit predict` covers (D76).
//!
//! Every mode follows one contract. A stream the fit bound data to is
//! _eligible_; a stream with no bound data that nobody asked for is _skipped_,
//! named with the reason `no_bound_data`, and does not affect the exit status,
//! because it was never eligible. A stream asked for by `--stream` that cannot
//! be produced is a failure, and so the exit status is 1 while every stream that
//! was produced is still written. When no stream is eligible at all there is
//! nothing to produce, and the command fails before any work.
//!
//! | situation                                                    | exit |
//! | ------------------------------------------------------------ | ---- |
//! | every eligible stream produced                               | 0    |
//! | every eligible stream produced, streams with no data skipped | 0    |
//! | a stream with no bound data asked for by `--stream`          | 1    |
//! | several streams asked for and any one cannot be produced     | 1    |
//! | no stream eligible                                           | 1    |
//! | an eligible stream fails while it is produced                | 1    |

use super::failures::{DeterministicFailure, SkipReason, SkippedStream, Site};
use ir::observation::ObservationModel;

/// The streams one `fit predict` covers, and the ones it leaves out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StreamSelection {
    /// The eligible leaves, by name, in the model's order.
    eligible: Vec<String>,
    /// The leaves `--stream` named; empty when it was not given.
    requested: Vec<String>,
    /// Streams left out without failing the run.
    pub skipped: Vec<SkippedStream>,
    /// Streams asked for by `--stream` that cannot be produced.
    pub failures: Vec<DeterministicFailure>,
}

impl StreamSelection {
    /// Whether the leaf `name` is produced.
    pub(crate) fn includes(&self, name: &str) -> bool {
        self.eligible.iter().any(|e| e == name)
    }

    /// Whether `--stream` named the leaf `name`, by its own name or its
    /// logical stream's.
    pub(crate) fn was_requested(&self, name: &str) -> bool {
        self.requested.iter().any(|r| r == name)
    }
}

/// Classify the model's observation leaves under the D76 contract.
///
/// `bound` says whether the fit bound data to a leaf's source. `requested` is
/// `--stream`, each entry a logical stream name (which selects all of its
/// leaves) or a leaf name. `site` names a requested stream's failure in the
/// caller's mode. Refused outright: a requested name the model does not
/// declare, and a selection with no eligible stream.
pub(crate) fn select_streams(
    observations: &[ObservationModel],
    bound: impl Fn(&ObservationModel) -> bool,
    requested: &[String],
    site: fn(String) -> Site,
) -> Result<StreamSelection, String> {
    let matches = |o: &ObservationModel, r: &str| o.source == r || o.name == r;
    for r in requested {
        if !observations.iter().any(|o| matches(o, r)) {
            let mut declared: Vec<&str> = observations.iter().map(|o| o.source.as_str()).collect();
            declared.dedup();
            return Err(format!(
                "--stream {r:?} names no observation stream this model declares (it declares: {})",
                declared.join(", ")));
        }
    }
    let mut out = StreamSelection {
        eligible: Vec::new(),
        requested: Vec::new(),
        skipped: Vec::new(),
        failures: Vec::new(),
    };
    for o in observations {
        let asked = requested.iter().any(|r| matches(o, r));
        if !requested.is_empty() && !asked {
            continue;
        }
        if asked {
            out.requested.push(o.name.clone());
        }
        match (bound(o), asked) {
            (true, _) => out.eligible.push(o.name.clone()),
            (false, false) => out.skipped.push(SkippedStream {
                stream: o.name.clone(),
                reason: SkipReason::NoBoundData,
            }),
            (false, true) => out.failures.push(DeterministicFailure::EvaluationFailed {
                at: site(o.name.clone()),
                reason: format!(
                    "--stream asked for it, but the fit bound no data to its source '{}', so \
                     there are no observed rows to predict it from. Bind a file to '{}' in the \
                     fit's [data.observations] and re-run the fit, or drop it from --stream.",
                    o.source, o.source),
            }),
        }
    }
    if out.eligible.is_empty() {
        let mut why: Vec<String> = out.failures.iter().map(|f| f.describe()).collect();
        why.extend(out.skipped.iter().map(|s| s.describe()));
        return Err(format!("no observation stream can be predicted:\n  {}", why.join("\n  ")));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::observation::*;

    fn leaf(name: &str, source: &str) -> ObservationModel {
        ObservationModel {
            name: name.into(),
            source: source.into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: source.into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: source.into(),
            emit_schedule: None,
            stratum: vec![],
            covers: Some(Covers::Until { offset: 0.0, span: 1.0 }),
            projection: Projection::CumulativeFlow("infection".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    fn model() -> Vec<ObservationModel> {
        vec![leaf("daily", "daily"), leaf("weekly", "weekly")]
    }

    fn daily_only(o: &ObservationModel) -> bool {
        o.source == "daily"
    }

    fn names(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }

    /// Row 2 of the D76 table: an unbound stream nobody asked for is skipped
    /// with `no_bound_data`, not refused, and nothing fails.
    #[test]
    fn an_unbound_stream_not_asked_for_is_skipped() {
        let s = select_streams(&model(), daily_only, &[], Site::Stream).unwrap();
        assert_eq!(names(&s.eligible), ["daily"]);
        assert_eq!(s.skipped, vec![SkippedStream { stream: "weekly".into(), reason: SkipReason::NoBoundData }]);
        assert!(s.failures.is_empty());
    }

    /// Row 4: two streams asked for, one with no data: the bound one is
    /// produced and the other is a failure, sited in the caller's mode.
    #[test]
    fn an_unbound_stream_asked_for_beside_a_bound_one_fails_alone() {
        let req = ["daily".to_string(), "weekly".to_string()];
        let s = select_streams(&model(), daily_only, &req, Site::ForecastStream).unwrap();
        assert_eq!(names(&s.eligible), ["daily"]);
        assert!(s.skipped.is_empty());
        assert!(matches!(&s.failures[..], [DeterministicFailure::EvaluationFailed {
            at: Site::ForecastStream(w), .. }] if w == "weekly"), "{:?}", s.failures);
        assert!(s.was_requested("weekly") && s.was_requested("daily"));
    }

    /// Rows 3 and 5: with no eligible stream left, the selection is refused,
    /// naming why each stream is out.
    #[test]
    fn d76_row5_no_eligible_stream_is_refused() {
        let err = select_streams(&model(), daily_only, &["weekly".to_string()], Site::Stream).unwrap_err();
        assert!(err.contains("stream `weekly`") && err.contains("bound no data"), "{err}");
        let err = select_streams(&model(), |_| false, &[], Site::Stream).unwrap_err();
        assert!(err.contains("skipped stream `daily` (no_bound_data)"), "{err}");
        assert!(err.contains("skipped stream `weekly` (no_bound_data)"), "{err}");
    }

    /// A logical name selects every leaf of its stream; a name the model does
    /// not declare is refused, naming the streams it does.
    #[test]
    fn a_logical_name_selects_its_leaves_and_an_unknown_name_is_refused() {
        let obs = vec![leaf("cases_a", "cases"), leaf("cases_b", "cases"), leaf("deaths", "deaths")];
        let s = select_streams(&obs, |_| true, &["cases".to_string()], Site::Stream).unwrap();
        assert_eq!(names(&s.eligible), ["cases_a", "cases_b"]);
        let err = select_streams(&obs, |_| true, &["hosp".to_string()], Site::Stream).unwrap_err();
        assert!(err.contains("\"hosp\"") && err.contains("cases, deaths"), "{err}");
    }
}
