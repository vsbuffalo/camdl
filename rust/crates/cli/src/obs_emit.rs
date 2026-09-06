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
//! outside the run and is dropped. That row was the zero-width bin the old
//! convention wrote as a count of zero at `t_start`; nothing scored it, and no
//! likelihood moves. An undeclared stream keeps that row, and every other byte
//! of its output, exactly as before.

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
pub(crate) fn plan_emission(
    obs: &ObservationModel,
    emit_times: &[f64],
    t_start: f64,
    t_end: f64,
) -> Result<EmitPlan, String> {
    let columns = temporal_columns(obs)?;
    let scored = obs.scored.clone();
    let eps = crate::OBS_SNAP_EPS;
    let within_run = |start: f64, stop: f64| start >= t_start - eps && stop <= t_end + eps;

    let rows: Vec<EmitRow> = match (obs.projection.temporal_kind(), &obs.covers) {
        // A state read at each instant; a declaration on it does not compile.
        (TemporalKind::Instant, _) => emit_times
            .iter()
            .map(|&t| EmitRow { label: t, coverage: Coverage::Instant })
            .collect(),
        // Undeclared (transitional): today's reading exactly — each row is the
        // flow since the previous emit time, the first since `t_start`, which
        // makes the first row zero-width when the schedule starts there.
        (TemporalKind::Interval, None) => {
            let mut prev = t_start;
            emit_times
                .iter()
                .map(|&t| {
                    let row = EmitRow { label: t, coverage: Coverage::Interval { start: prev, stop: t } };
                    prev = t;
                    row
                })
                .collect()
        }
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
                if within_run(start, stop) {
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
                if t > prev + eps {
                    rows.push(EmitRow { label: t, coverage: Coverage::Interval { start: prev, stop: t } });
                }
                prev = t;
            }
            rows
        }
    };
    Ok(EmitPlan { columns, scored, rows })
}

/// The stream's declared temporal anchor: exactly one of a `: time` column or a
/// window pair (the compiler enforces this; the error here guards a malformed
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

    #[test]
    fn an_undeclared_stream_keeps_the_old_rows_including_the_zero_width_first() {
        let s = stream(incidence(), None, time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0).unwrap();
        assert_eq!(plan.columns, TemporalColumns::Time("day".into()));
        assert_eq!(plan.scored, "n_cases", "the value goes under the SCORED column, not the stream name");
        assert_eq!(plan.coverages(), vec![(0.0, iv(0.0, 0.0)), (7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))]);
    }

    #[test]
    fn closing_at_drops_the_leading_row_whose_period_precedes_the_run() {
        // closing_at(day, 7): [t−7, t). The emit time 0 would cover [−7, 0),
        // which the run never simulated; 7 and 14 cover [0,7) and [7,14).
        let s = stream(incidence(), Some(Covers::Until { offset: 0.0, span: 7.0 }), time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0).unwrap();
        assert_eq!(plan.coverages(), vec![(7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))]);
    }

    #[test]
    fn day_drops_the_trailing_row_whose_period_runs_past_the_horizon() {
        // day(t): [t, t+1). Labels 3 and 4 fit inside a run ending at 5; the
        // label 5 would cover [5, 6), past the horizon.
        let s = stream(incidence(), Some(Covers::From { offset: 0.0, span: 1.0 }), time_cols());
        let plan = plan_emission(&s, &[3.0, 4.0, 5.0], 0.0, 5.0).unwrap();
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
        let plan = plan_emission(&s, &[0.0, 7.0, 14.0], 0.0, 14.0).unwrap();
        assert_eq!(plan.columns, TemporalColumns::Window { start: "from".into(), stop: "until".into() });
        assert_eq!(plan.coverages(), vec![(7.0, iv(0.0, 7.0)), (14.0, iv(7.0, 14.0))],
            "the emit time at t_start would be a zero-width window and is not a row");
    }

    #[test]
    fn an_instant_stream_emits_every_time_with_no_span() {
        let s = stream(Projection::CurrentPop("I".into()), None, time_cols());
        let plan = plan_emission(&s, &[0.0, 7.0], 0.0, 7.0).unwrap();
        assert_eq!(plan.coverages(), vec![(0.0, Coverage::Instant), (7.0, Coverage::Instant)]);
    }
}
