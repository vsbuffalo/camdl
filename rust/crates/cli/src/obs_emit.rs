//! What `simulate --obs` writes for one observation stream: which rows, under
//! which temporal columns, each drawn over which period (gh#833, ruling 3 of
//! the observation-time proposal).
//!
//! The emitter follows the stream's own declaration, so the file it writes is
//! the file the loader reads back. The label is the emit-schedule time; the
//! value is the flow over whatever period the stream's `covers` assigns to
//! that label — the same arithmetic the loader applies to a label it reads
//! (`Covers::period_of`) — and a row whose period does not lie within the run
//! is not written, because the run never simulated it. A stream declared with
//! `window_start`/`window_stop` gets contiguous windows closing at the emit
//! times, written under the declared column names.
//!
//! One consequence is visible: under `closing_at` with a schedule starting at
//! `t_start`, the first emit time's period `[t_start − Δ, t_start)` falls
//! outside the run and is dropped — a row nothing could score, so no likelihood
//! moves. An accumulating stream whose IR carries no declaration has no reading
//! to write under and is refused (the compiler never produces one, E350).

use ir::observation::{ColumnRole, Covers, ObservationModel, TemporalKind};
use sim::inference::Coverage;

/// The temporal column(s) an emitted file carries for a stream — the ones the
/// stream declared in `columns { }`, so the file re-loads under its own model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TemporalColumns {
    /// One label column, under the stream's `: time` column name.
    Time(String),
    /// A `window_start` / `window_stop` pair, under their declared names.
    Window { start: String, stop: String },
}

/// One row the emitter writes: what its temporal column(s) say, and what the
/// value is drawn over.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EmitRow {
    /// The `: time` column's value — the emit-schedule time. For a windowed
    /// stream this is the window's stop; the start is in `coverage`.
    pub label: f64,
    /// An instant at `label`, or the flow over `[start, stop)`.
    pub coverage: Coverage,
}

/// Everything a writer needs to emit one stream's file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EmitPlan {
    pub columns: TemporalColumns,
    /// The scored column's name — the header the value goes under (gh#830:
    /// it was the stream's name, which is not what the loader reads).
    pub scored: String,
    pub rows: Vec<EmitRow>,
}

impl EmitPlan {
    pub fn labels(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.label).collect()
    }

    pub fn coverages(&self) -> Vec<(f64, Coverage)> {
        self.rows.iter().map(|r| (r.label, r.coverage)).collect()
    }
}

/// Plan the rows for `obs` at the emit-schedule times `emit_times`, within the
/// run `[t_start, t_end]`. `emit_times` are already sorted, on the run's grid,
/// and at most `t_end` (`obs_emit_schedule_times`).
///
/// `override_step` is the `--emit-every` cadence for this stream when the flag
/// names it (gh#656), in axis units. A uniform `covers` form's span is then
/// re-widened to that cadence: the rows are re-spaced, and a row still covers
/// the whole span back to its neighbour — a daily stream emitted weekly writes
/// weekly totals, not one day in seven. Where the label sits on its window is
/// kept.
pub(crate) fn plan_emission(
    obs: &ObservationModel,
    emit_times: &[f64],
    t_start: f64,
    t_end: f64,
    override_step: Option<f64>,
) -> Result<EmitPlan, String> {
    let columns = temporal_columns(obs)?;
    let scored = obs.scored.clone();
    let eps = crate::OBS_SNAP_EPS;
    let within_run = |start: f64, stop: f64| start >= t_start - eps && stop <= t_end + eps;
    // A LABEL outside the run is a declared emission the run cannot produce
    // (an `at [...]` time before `simulate.from`, gh#589), which stays the
    // projection's hard error — such a row is kept here so the guard sees it.
    // Only a row whose label the run does reach, but whose period spills past
    // an end of it, is quietly not written.
    let label_in_run = |t: f64| t >= t_start - eps && t <= t_end + eps;

    let covers: Option<Covers> = match (&obs.covers, override_step) {
        (Some(Covers::From { offset, .. }), Some(step)) => Some(Covers::From { offset: *offset, span: step }),
        (Some(Covers::Until { offset, .. }), Some(step)) => Some(Covers::Until { offset: *offset, span: step }),
        (c, _) => c.clone(),
    };

    let rows: Vec<EmitRow> = match (obs.projection.temporal_kind(), &covers) {
        // A state read at each instant; a declaration on it does not compile.
        (TemporalKind::Instant, _) => emit_times
            .iter()
            .map(|&t| EmitRow { label: t, coverage: Coverage::Instant })
            .collect(),
        // An accumulating stream with no declaration is an IR the compiler
        // cannot have produced (E350); there is no reading to write under.
        (TemporalKind::Interval, None) => return Err(crate::pfilter::missing_covers_error(obs)),
        // A uniform form: the label's period, kept only when the run covers it.
        (TemporalKind::Interval, Some(covers @ (Covers::From { .. } | Covers::Until { .. }))) => {
            let mut rows = Vec::with_capacity(emit_times.len());
            for &t in emit_times {
                let (start, stop) = covers.period_of(t)
                    .expect("a uniform form assigns every label a period");
                if !(stop > start) {
                    return Err(format!(
                        "observation stream '{}': `covers` gives the row labelled {} the \
                         empty period [{}, {}) — its width must be positive",
                        obs.name, t, start, stop
                    ));
                }
                if !label_in_run(t) || within_run(start, stop) {
                    rows.push(EmitRow { label: t, coverage: Coverage::Interval { start, stop } });
                }
            }
            rows
        }
        // Per-row windows: contiguous, closing at each emit time. The first
        // opens at `t_start`; an emit time AT `t_start` would be a zero-width
        // window, which is not a period, and is not written.
        (TemporalKind::Interval, Some(Covers::WindowColumns)) => {
            let mut prev = t_start;
            let mut rows = Vec::with_capacity(emit_times.len());
            for &t in emit_times {
                if t > prev + eps || !label_in_run(t) {
                    rows.push(EmitRow { label: t, coverage: Coverage::Interval { start: prev, stop: t } });
                }
                prev = t;
            }
            rows
        }
    };
    Ok(EmitPlan { columns, scored, rows })
}

/// The stream's declared temporal columns: exactly one of a `: time` column or
/// a window pair (the compiler enforces this; the error here guards a malformed
/// IR).
fn temporal_columns(obs: &ObservationModel) -> Result<TemporalColumns, String> {
    let start = crate::pfilter::column_with_role(obs, &ColumnRole::WindowStart);
    let stop = crate::pfilter::column_with_role(obs, &ColumnRole::WindowStop);
    match (start, stop) {
        (Some(start), Some(stop)) => Ok(TemporalColumns::Window {
            start: start.to_string(),
            stop: stop.to_string(),
        }),
        _ => crate::pfilter::obs_time_column(obs).map(|c| TemporalColumns::Time(c.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::observation::{Likelihood, ObsColumn, ObservationSchedule, PoissonLikelihood, Projection};

    fn stream(projection: Projection, covers: Option<Covers>, columns: Vec<ObsColumn>) -> ObservationModel {
        ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns,
            scored: "n_cases".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            covers,
            projection,
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    fn time_cols() -> Vec<ObsColumn> {
        vec![
            ObsColumn { name: "day".into(), role: ColumnRole::Time },
            ObsColumn { name: "n_cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ]
    }

    fn iv(start: f64, stop: f64) -> Coverage {
        Coverage::Interval { start, stop }
    }

    fn incidence() -> Projection {
        Projection::CumulativeFlow("infection".into())
    }

    /// An accumulating stream whose IR says nothing about what its rows cover
    /// has no reading to write under: there is no default period, so the plan
    /// refuses rather than emit rows under a convention the model never stated.
    #[test]
    fn an_interval_stream_with_no_declaration_is_refused() {
        let s = stream(incidence(), None, time_cols());
        let e = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0, None).unwrap_err();
        assert!(e.contains("cases") && e.contains("E350"), "names the stream and the rule: {e}");
    }

    #[test]
    fn closing_at_drops_the_leading_row_whose_period_precedes_the_run() {
        // closing_at(day, 7): [t−7, t). The emit time 0 would cover [−7, 0),
        // which the run never simulated; 7 and 14 cover [0,7) and [7,14).
        let s = stream(incidence(), Some(Covers::Until { offset: 0.0, span: 7.0 }), time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0, None).unwrap();
        assert_eq!(plan.columns, TemporalColumns::Time("day".into()));
        assert_eq!(plan.scored, "n_cases", "the value goes under the SCORED column, not the stream name");
        assert_eq!(plan.coverages(), vec![(7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))]);
    }

    #[test]
    fn day_drops_the_trailing_row_whose_period_runs_past_the_horizon() {
        // day(t): [t, t+1). Labels 3 and 4 fit inside a run ending at 5; the
        // label 5 would cover [5, 6), past the horizon.
        let s = stream(incidence(), Some(Covers::From { offset: 0.0, span: 1.0 }), time_cols());
        let plan = plan_emission(&s, &[3.0, 4.0, 5.0], 0.0, 5.0, None).unwrap();
        assert_eq!(plan.coverages(), vec![(3.0, iv(3.0, 4.0)), (4.0, iv(4.0, 5.0))]);
    }

    #[test]
    fn window_columns_are_contiguous_and_named_as_declared() {
        let cols = vec![
            ObsColumn { name: "from".into(), role: ColumnRole::WindowStart },
            ObsColumn { name: "until".into(), role: ColumnRole::WindowStop },
            ObsColumn { name: "n_cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ];
        let s = stream(incidence(), Some(Covers::WindowColumns), cols);
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0, None).unwrap();
        assert_eq!(plan.columns, TemporalColumns::Window { start: "from".into(), stop: "until".into() });
        assert_eq!(plan.coverages(), vec![(7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))],
            "the emit time at t_start would be a zero-width window and is not a row");
    }

    /// gh#656 × gh#833: `--emit-every 7` on a stream declared
    /// `closing_at(day, 1 'days)` re-spaces the rows _and_ re-widens each
    /// window to the week it now spans — weekly totals, not one day in seven.
    /// Where the label sits is kept: a `closing_at` row still closes at its
    /// label.
    #[test]
    fn an_emit_every_override_rewidens_a_uniform_window_to_its_cadence() {
        let s = stream(incidence(), Some(Covers::Until { offset: 0.0, span: 1.0 }), time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0, Some(7.0)).unwrap();
        assert_eq!(plan.coverages(), vec![(7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))]);
        // `day(t)`: a row still opens at its label.
        let s = stream(incidence(), Some(Covers::From { offset: 0.0, span: 1.0 }), time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0], 0.0, 14.0, Some(7.0)).unwrap();
        assert_eq!(plan.coverages(), vec![(0.0, iv(0.0, 7.0)), (7.0, iv(7.0, 14.0))]);
    }

    /// gh#589 × gh#833: an emit time BEFORE the run's start is a declared
    /// emission the run cannot produce, not a zero-width leading row. It stays
    /// in the plan so the projection's guard refuses it by name, rather than
    /// vanishing quietly with the row the run merely does not cover.
    #[test]
    fn a_label_outside_the_run_is_kept_for_the_guard_to_refuse() {
        let s = stream(incidence(), Some(Covers::Until { offset: 0.0, span: 20.0 }), time_cols());
        // The run starts at 10; the list declares 0. The label 20 is inside the
        // run but its period [0, 20) is not, so that row is simply not written.
        let plan = plan_emission(&s, &[0.0, 20.0, 40.0], 10.0, 40.0, None).unwrap();
        assert_eq!(plan.labels(), vec![0.0, 40.0], "the out-of-window label survives");
        // Whereas a label AT the start whose period precedes the run is dropped.
        let plan = plan_emission(&s, &[10.0, 30.0], 10.0, 40.0, None).unwrap();
        assert_eq!(plan.labels(), vec![30.0]);
    }

    #[test]
    fn an_instant_stream_emits_every_time_with_no_span() {
        let s = stream(Projection::CurrentPop("I".into()), None, time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0], 0.0, 7.0, None).unwrap();
        assert_eq!(plan.coverages(), vec![(0.0, Coverage::Instant), (7.0, Coverage::Instant)]);
    }
}
