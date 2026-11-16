//! The one reader of an observation stream's rows, and what each row measures
//! (`docs/dev/temporal-architecture.md` §2.2, §3.1).
//!
//! [`observed_rows`] reads one leaf's rows from its data file once, through the
//! column readers in `pfilter`, and records for each row the instant or period
//! it measures. The likelihood loader (`runner::load_observations`) and the
//! stream's periods (`pfilter::stream_times_of_rows`) are both built from these
//! rows, so what the likelihood scores and where it scores it cannot come from
//! two different reads of the file.
//!
//! The rows are those of the file, routed to the leaf, before any filtering
//! that depends on the run: the drop of a hole whose period opens before the
//! run starts is applied by `load_observations` afterwards, and whether a
//! period's boundaries may lie off the integration grid is the method's to
//! say once it is known (`runner::refuse_periods_off_the_grid`).

use ir::observation::{Covers, ObservationModel, TemporalKind};
use sim::inference::{ObsCell, Period};

use crate::caltime_load::{TemporalCell, TimeOpts};

/// What one row measures: a state read at an instant, or a count accumulated
/// over a period. The end is when the row's information is complete, and is
/// the time it is scored at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Measures {
    Instant(f64),
    Period(Period),
}

impl Measures {
    /// The instant, or the period's start.
    pub(crate) fn start(&self) -> f64 {
        match self {
            Measures::Instant(t) => *t,
            Measures::Period(p) => p.start(),
        }
    }

    /// The instant, or the period's stop: the time the row is scored at.
    pub(crate) fn end(&self) -> f64 {
        match self {
            Measures::Instant(t) => *t,
            Measures::Period(p) => p.stop(),
        }
    }
}

/// One data row of one stream (leaf), with what it measures.
#[derive(Debug, Clone)]
pub(crate) struct ObservedRow {
    /// The row's index among this leaf's rows, in file order.
    pub position: usize,
    /// The instant or period the row measures, from the stream's `covers`
    /// declaration or its window columns.
    pub measures: Measures,
    /// The row's time cells as the file wrote them: `[window_start, time]` for a
    /// stream with per-row windows (where the time column is `window_stop`),
    /// `[time]` otherwise.
    #[cfg_attr(not(test), expect(dead_code, reason = "foundation for the temporal-architecture \
        migration; read by the row digest"))]
    pub temporal: Vec<TemporalCell>,
    /// The observed value; `None` is an `NA` row (a hole), which keeps its
    /// period.
    pub cell: Option<ObsCell>,
    /// The auxiliary values the likelihood reads (a binomial denominator, an
    /// offset). Empty for a hole.
    pub aux: Vec<(String, f64)>,
    /// The row's time cell converted to model time, before `covers` gives it a
    /// period. Private: only the label-keyed sites that Stage 2 of the
    /// temporal-architecture migration replaces read it, through
    /// [`ObservedRow::label`], and Stage 2 removes it.
    label: f64,
}

impl ObservedRow {
    /// The row's time cell in model time — not what the row measures. Read by
    /// the sites that still compare labels (the holdout cut, observation
    /// anchors) until Stage 2 of the
    /// temporal-architecture migration moves them onto [`Self::measures`] and
    /// removes this accessor.
    pub(crate) fn label(&self) -> f64 {
        self.label
    }
}

/// Read one leaf's rows from `path`: its labels, the cells as written, the
/// observed values with holes, the auxiliary values, and what each row
/// measures.
///
/// `siblings` is every observation block sharing `obs`'s source; a long-form
/// (stratified) file routes its rows to `obs` by stratum among them.
pub(crate) fn observed_rows(
    path: &str,
    obs: &ObservationModel,
    siblings: &[&ObservationModel],
    opts: &TimeOpts,
) -> Result<Vec<ObservedRow>, String> {
    let (labels, written, cells, mut aux) = if crate::pfilter::is_long_form_stream(obs) {
        crate::pfilter::load_long_form_stream_as_written(path, obs, siblings, opts)?
    } else {
        // Bind the file columns by name: the declared `Time`-role column is the
        // time axis, and `scored` is the value column.
        let time_col = crate::pfilter::obs_time_column(obs)?;
        let value_col = &obs.scored;
        let domain = crate::pfilter::CellDomain::of(obs, value_col);
        let (times, written, mut cells) = crate::pfilter::load_data_tsv_column_cells_as_written(
            path, time_col, value_col, domain, opts)?;
        // Per-observation auxiliary data (binomial `n = tested`, person-time
        // offset). A row where the scored value or any referenced aux is `NA`
        // is a hole (present-together-or-hole).
        let aux_cols: Vec<(String, crate::pfilter::CellDomain)> =
            crate::pfilter::stream_aux_columns(obs).into_iter()
                .map(|c| { let d = crate::pfilter::CellDomain::of(obs, &c); (c, d) })
                .collect();
        let (aux, force_hole) = crate::pfilter::load_stream_aux(path, &aux_cols, cells.len())?;
        for (cell, hole) in cells.iter_mut().zip(&force_hole) {
            if *hole {
                *cell = None;
            }
        }
        (times, written, cells, aux)
    };
    for (cell, row_aux) in cells.iter().zip(aux.iter_mut()) {
        if cell.is_none() {
            row_aux.clear();
        }
    }

    let window_starts = crate::pfilter::load_window_starts_if_windowed(obs, path, opts)?;
    let measures = measures_of(
        obs,
        obs.projection.temporal_kind(),
        &labels,
        window_starts.as_ref().map(|(starts, _)| starts.as_slice()),
        path,
    )?;

    let mut start_written = window_starts.map(|(_, written)| written.into_iter());
    Ok(labels.into_iter()
        .zip(written)
        .zip(cells.into_iter().zip(aux))
        .zip(measures)
        .enumerate()
        .map(|(position, (((label, written), (cell, aux)), measures))| {
            // `measures_of` refused a window column whose length differs from
            // the time column's, so a start exists for every row.
            let temporal = match start_written.as_mut().and_then(Iterator::next) {
                Some(start) => vec![start, written],
                None => vec![written],
            };
            ObservedRow { position, measures, temporal, cell, aux, label }
        })
        .collect())
}

/// What each row of `obs` measures, given the rows' labels (their time cells in
/// model time) and, for a stream with per-row windows, its `window_start`
/// cells in model time.
///
/// A stream with no `covers` declaration is a state read at each label. A
/// uniform `covers` form assigns each label the period [`Covers::period_of`]
/// computes. Per-row windows run from each row's `window_start` to its label,
/// which for such a stream is `window_stop`. An accumulating stream with no
/// declaration is an IR the compiler cannot produce (E350) and is refused.
///
/// The one lowering from labels to what rows measure: [`observed_rows`] and
/// `pfilter::stream_times_for` (the lowering tests' view, handed labels
/// directly) both call it.
pub(crate) fn measures_of(
    obs: &ObservationModel,
    kind: TemporalKind,
    labels: &[f64],
    window_starts: Option<&[f64]>,
    path: &str,
) -> Result<Vec<Measures>, String> {
    let Some(covers) = &obs.covers else {
        return match kind {
            TemporalKind::Instant => Ok(labels.iter().map(|&t| Measures::Instant(t)).collect()),
            TemporalKind::Interval => Err(crate::pfilter::missing_covers_error(obs)),
        };
    };

    let periods = match covers {
        Covers::From { .. } | Covers::Until { .. } => {
            labels.iter().map(|&t| {
                let (start, stop) = covers.period_of(t)
                    .expect("a uniform form assigns every label a period");
                Period::new(start, stop)
            }).collect::<Result<Vec<_>, _>>()
        }
        Covers::WindowColumns => {
            let starts = window_starts
                .expect("the caller reads the window starts of a stream with per-row windows");
            if starts.len() != labels.len() {
                let start_col = crate::pfilter::column_with_role(
                    obs, &ir::observation::ColumnRole::WindowStart).unwrap_or("window_start");
                return Err(format!(
                    "observation stream '{}': '{}' has {} data rows but the window_stop \
                     column has {}", obs.name, start_col, starts.len(), labels.len()));
            }
            starts.iter().zip(labels).map(|(&a, &b)| Period::new(a, b))
                .collect::<Result<Vec<_>, _>>()
        }
    };

    // `Period::new` reports one bad row; name the stream so the message says
    // which file to look in.
    let periods = periods.map_err(|e| format!(
        "observation stream '{}' ({}): {}", obs.name, path, e))?;
    Ok(periods.into_iter().map(Measures::Period).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caltime_load::TimeFormat;
    use ir::observation::*;

    fn numeric_opts() -> TimeOpts<'static> {
        TimeOpts { origin: None, time_unit: "days", dt: 1.0, t_start: 0.0, format: TimeFormat::Auto }
    }

    fn write_tsv(name: &str, content: &str) -> String {
        let dir = crate::test_support::unique_temp_dir(&format!("observed_rows_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.tsv");
        std::fs::write(&path, content).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn poisson() -> Likelihood {
        Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
        })
    }

    fn stream(columns: Vec<ObsColumn>, covers: Option<Covers>, projection: Projection)
        -> ObservationModel
    {
        ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns,
            scored: "cases".into(),
            emit_schedule: None,
            stratum: vec![],
            covers,
            projection,
            projection_state_grad: Default::default(),
            likelihood: poisson(),
        }
    }

    fn timed_columns() -> Vec<ObsColumn> {
        vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ]
    }

    fn incidence(covers: Covers) -> ObservationModel {
        stream(timed_columns(), Some(covers), Projection::CumulativeFlow("infection".into()))
    }

    fn periods(rows: &[ObservedRow]) -> Vec<(f64, f64)> {
        rows.iter().map(|r| match r.measures {
            Measures::Period(p) => (p.start(), p.stop()),
            Measures::Instant(t) => panic!("expected a period, got the instant {t}"),
        }).collect()
    }

    /// The spec's §2.2 table, one row per uniform `covers` form, each in the
    /// lowered form the expander emits (`ir::observation::Covers`): the row
    /// labelled 21 measures the period the table gives, and the next row,
    /// labelled 28, the period one week on.
    #[test]
    fn each_uniform_covers_form_gives_the_periods_of_the_spec_table() {
        let path = write_tsv("uniform", "time\tcases\n21\t5\n28\t6\n");
        let cases = [
            ("day(time)", Covers::From { offset: 0.0, span: 1.0 }, [(21.0, 22.0), (28.0, 29.0)]),
            ("starting_on(time, 7 'days)", Covers::From { offset: 0.0, span: 7.0 },
                [(21.0, 28.0), (28.0, 35.0)]),
            ("ending_on(time, 7 'days)", Covers::Until { offset: 1.0, span: 7.0 },
                [(15.0, 22.0), (22.0, 29.0)]),
            ("closing_at(time, 7 'days)", Covers::Until { offset: 0.0, span: 7.0 },
                [(14.0, 21.0), (21.0, 28.0)]),
        ];
        for (form, covers, expected) in cases {
            let obs = incidence(covers);
            let rows = observed_rows(&path, &obs, &[&obs], &numeric_opts())
                .unwrap_or_else(|e| panic!("{form}: {e}"));
            assert_eq!(periods(&rows), expected.to_vec(), "{form}");
            assert_eq!(rows.iter().map(|r| r.measures.end()).collect::<Vec<_>>(),
                expected.iter().map(|&(_, stop)| stop).collect::<Vec<_>>(),
                "{form}: a period row ends at its period's stop");
            assert_eq!(rows.iter().map(|r| r.label()).collect::<Vec<_>>(), vec![21.0, 28.0],
                "{form}: the label is the time cell, whatever the form");
            assert_eq!(rows.iter().map(|r| r.position).collect::<Vec<_>>(), vec![0, 1]);
            assert_eq!(rows[0].temporal, vec![TemporalCell::Number(21.0)], "{form}");
            assert_eq!(rows[0].cell, Some(ObsCell::Scalar(5.0)), "{form}");
        }
    }

    /// The last row of the table: per-row windows are read from the two
    /// columns, and both cells are kept as written, start first. A gap between
    /// rows is a statement under this form, so the second row may open after
    /// the first closes.
    #[test]
    fn window_columns_give_the_periods_the_columns_state() {
        let path = write_tsv("windows",
            "window_start\twindow_stop\tcases\n14\t21\t5\n24\t28\t6\n");
        let obs = stream(
            vec![
                ObsColumn { name: "window_start".into(), role: ColumnRole::WindowStart },
                ObsColumn { name: "window_stop".into(), role: ColumnRole::WindowStop },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            Some(Covers::WindowColumns),
            Projection::CumulativeFlow("infection".into()),
        );
        let rows = observed_rows(&path, &obs, &[&obs], &numeric_opts()).unwrap();
        assert_eq!(periods(&rows), vec![(14.0, 21.0), (24.0, 28.0)]);
        assert_eq!(rows[1].temporal, vec![TemporalCell::Number(24.0), TemporalCell::Number(28.0)]);
        assert_eq!(rows[1].label(), 28.0, "the label of a windowed row is its window_stop");
    }

    /// A state read (no `covers`, a prevalence projection) measures the
    /// instant of its label.
    #[test]
    fn a_state_stream_measures_instants() {
        let path = write_tsv("instants", "time\tcases\n21\t5\n28\t6\n");
        let obs = stream(timed_columns(), None, Projection::CurrentPop("I".into()));
        let rows = observed_rows(&path, &obs, &[&obs], &numeric_opts()).unwrap();
        assert_eq!(rows.iter().map(|r| r.measures).collect::<Vec<_>>(),
            vec![Measures::Instant(21.0), Measures::Instant(28.0)]);
        assert_eq!(rows[0].measures.start(), 21.0);
        assert_eq!(rows[0].measures.end(), 21.0);
    }

    /// An `NA` row is a hole: no value, no auxiliary values, and the period
    /// its label assigns, like any other row.
    #[test]
    fn an_na_row_keeps_its_period_with_no_cell() {
        let path = write_tsv("na", "time\tcases\n21\t5\n28\tNA\n35\t7\n");
        let obs = incidence(Covers::Until { offset: 0.0, span: 7.0 });
        let rows = observed_rows(&path, &obs, &[&obs], &numeric_opts()).unwrap();
        assert_eq!(rows.iter().map(|r| r.cell).collect::<Vec<_>>(),
            vec![Some(ObsCell::Scalar(5.0)), None, Some(ObsCell::Scalar(7.0))]);
        assert_eq!(periods(&rows), vec![(14.0, 21.0), (21.0, 28.0), (28.0, 35.0)]);
        assert!(rows[1].aux.is_empty());
    }

    /// A dated time column is kept as the text the file wrote, while the
    /// period is computed from the converted time.
    #[test]
    fn a_dated_time_cell_is_kept_as_written() {
        let path = write_tsv("dated", "time\tcases\n2026-07-22\t5\n");
        let opts = TimeOpts { origin: Some("2026-07-01"), ..numeric_opts() };
        let obs = incidence(Covers::From { offset: 0.0, span: 1.0 });
        let rows = observed_rows(&path, &obs, &[&obs], &opts).unwrap();
        assert_eq!(rows[0].temporal, vec![TemporalCell::Date("2026-07-22".into())]);
        assert_eq!(periods(&rows), vec![(21.0, 22.0)]);
    }
}
