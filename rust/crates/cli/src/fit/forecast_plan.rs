//! What a fixed-origin forecast predicts (`docs/dev/temporal-architecture.md`
//! §2.5, §3.3, §4.3).
//!
//! A [`ForecastPlan`] is built once, from the fit's training partition and each
//! stream's declaration, before any filtering. It is the only place a target
//! is made: the driver ([`super::forecast_driver`]) appends every target to its
//! stream as a declared period carrying no value and reads a stream's running
//! total only at the end of one of them. A period the plan did not declare is
//! never read, because a stream's running total resets only where a declared
//! period starts, so an undeclared one would silently include the previous
//! period's flow (§4.4).
//!
//! The forecast origin is the training extent and the state is conditioned on
//! the training rows only (D57); the horizon counts each stream's own targets
//! from the first one ending after the origin (§2.5, D41).

use super::observed_rows::{Measures, ObservedRow};
use super::runner::TIME_TIE_EPS;
use super::training_partition::TrainingPartition;
use ir::observation::{Covers, ObservationModel, ObservationSchedule, TemporalKind};
use sim::inference::Period;

/// One stream's period (or instant) the forecast predicts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Target {
    /// The leaf (observation block) name.
    pub stream: String,
    /// The period or instant predicted.
    pub measures: Measures,
    /// The position of this target among its stream's targets; 0 is the first
    /// that ends after the origin (the period under way, when one is).
    pub horizon: i64,
    /// The duration from the origin to the target's end, in model time units.
    pub lead: f64,
    /// The held-out row that measures the same period, when one exists.
    pub observed: Option<HeldOutRow>,
}

/// Which file a held-out row is in: the training file (rows past the
/// training prefix, under `holdout_after`) or the stream's `[data.holdout]`
/// file. Row positions are counted within a file, as everywhere in camdl, so a
/// position names a row only together with its file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowFile {
    Training,
    Holdout,
}

impl RowFile {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            RowFile::Training => "training",
            RowFile::Holdout => "holdout",
        }
    }
}

/// A held-out row's place: its file and its position within that file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeldOutRow {
    pub file: RowFile,
    pub position: usize,
}

/// A row after the training prefix, with the file it was read from.
#[derive(Debug, Clone)]
pub(crate) struct HeldOut {
    pub file: RowFile,
    pub row: ObservedRow,
}

/// Where a stream's targets come from (§4.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TargetSource {
    /// A uniform `covers` form: the stream's tiling continued at its width.
    DeclaredTiling { span: f64 },
    /// The stream's `emit_schedule`: the periods between consecutive schedule
    /// times (window columns) or the schedule's instants (a state read).
    EmitSchedule,
}

impl TargetSource {
    pub(crate) fn describe(&self) -> String {
        match self {
            TargetSource::DeclaredTiling { span } => format!("declared_tiling(span = {span})"),
            TargetSource::EmitSchedule => "emit_schedule".to_string(),
        }
    }
}

/// What one stream contributes to a plan.
pub(crate) struct PlanStream<'a> {
    pub obs: &'a ObservationModel,
    /// The rows the parameters were fitted to (the stream's training prefix).
    pub training: &'a [ObservedRow],
    /// Rows after the training prefix, from the same file or a
    /// `[data.holdout]` file; they only name a target's `observed` row.
    pub held_out: &'a [HeldOut],
}

/// Why a stream has no targets (§4.3, D42, D52).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StreamRefusal {
    /// A stream with window columns and no `emit_schedule`, or a state read
    /// with no `emit_schedule`, declares no rhythm to continue (D42).
    NoRhythm { stream: String, kind: TemporalKind },
    /// The stream binds no rows, so its tiling has no phase.
    NoBoundRows { stream: String },
    /// The stream's likelihood reads data columns, which have no value for a
    /// target past the data (D52).
    ReadsDataColumn { stream: String, columns: Vec<String> },
    /// A target computed from the declaration overlaps one of the stream's
    /// training rows.
    OverlapsObserved { stream: String, start: f64, end: f64, row: usize },
}

impl StreamRefusal {
    /// The refused stream's leaf name.
    pub(crate) fn stream(&self) -> &str {
        match self {
            StreamRefusal::NoRhythm { stream, .. }
            | StreamRefusal::NoBoundRows { stream }
            | StreamRefusal::ReadsDataColumn { stream, .. }
            | StreamRefusal::OverlapsObserved { stream, .. } => stream,
        }
    }
}

impl std::fmt::Display for StreamRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamRefusal::NoRhythm { stream, kind } => {
                let what = match kind {
                    TemporalKind::Interval => {
                        "its rows state their own windows (window_start / window_stop), so \
                         there is no rule that says which period comes next"
                    }
                    TemporalKind::Instant => {
                        "it reads the state at the instants its rows name, so there is no \
                         rule that says which instant comes next"
                    }
                };
                write!(f,
                    "stream '{stream}' is refused a forecast: {what}. The spacing between \
                     past rows is not a declaration. Declare the rhythm either with a \
                     uniform `covers` form (an interval stream, e.g. \
                     `covers = ending_on(time, 7 'days)`) or with `emit_schedule` \
                     (e.g. `emit_schedule = every 7 'days`).")
            }
            StreamRefusal::NoBoundRows { stream } => write!(f,
                "stream '{stream}' is refused a forecast: it binds no rows, so its periods \
                 have no phase to continue."),
            StreamRefusal::ReadsDataColumn { stream, columns } => write!(f,
                "stream '{stream}' is refused a forecast: its likelihood reads the data \
                 column(s) [{}], which have no value for a target past the data. Express \
                 the column in the model (a parameter or a compartment sum) to forecast \
                 this stream.", columns.join(", ")),
            StreamRefusal::OverlapsObserved { stream, start, end, row } => write!(f,
                "stream '{stream}' is refused a forecast: its declared rhythm puts a \
                 target at [{start}, {end}), which overlaps its training row {row}. The \
                 rhythm and the rows disagree about where the stream's periods fall."),
        }
    }
}

/// Why no plan can be built at all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PlanError {
    /// The horizon is not a positive finite duration.
    BadHorizon(f64),
    /// No stream has a target within the horizon. Carries each stream's
    /// refusal, if it had one.
    NoTargets { refusals: Vec<StreamRefusal> },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::BadHorizon(h) => write!(f,
                "the forecast horizon must be a positive duration; got {h}"),
            PlanError::NoTargets { refusals } => {
                write!(f, "no stream has a forecast target within the horizon")?;
                for r in refusals {
                    write!(f, "\n  {r}")?;
                }
                Ok(())
            }
        }
    }
}

/// What a fixed-origin forecast predicts.
#[derive(Debug, Clone)]
pub(crate) struct ForecastPlan {
    origin: f64,
    horizon: f64,
    parameters: TrainingPartition,
    /// Sorted by (end, stream).
    targets: Vec<Target>,
    /// Each stream that has targets, and where they came from.
    sources: Vec<(String, TargetSource)>,
    /// Each stream refused a forecast, and why.
    refusals: Vec<StreamRefusal>,
}

impl ForecastPlan {
    /// The plan for a forecast of reach `horizon` from the training extent of
    /// `parameters`, over `streams`. Each stream's targets come from its
    /// declaration (§4.3); a stream that declares none is refused and recorded,
    /// and the plan fails only when no stream has a target.
    pub(crate) fn new(
        parameters: TrainingPartition,
        horizon: f64,
        streams: &[PlanStream],
    ) -> Result<Self, PlanError> {
        if !(horizon.is_finite() && horizon > 0.0) {
            return Err(PlanError::BadHorizon(horizon));
        }
        let origin = parameters.extent();
        let mut targets: Vec<Target> = Vec::new();
        let mut sources = Vec::new();
        let mut refusals = Vec::new();
        for s in streams {
            match stream_targets(s, origin, horizon) {
                Ok((source, ts)) => {
                    if !ts.is_empty() {
                        sources.push((s.obs.name.clone(), source));
                        targets.extend(ts);
                    }
                }
                Err(r) => refusals.push(r),
            }
        }
        if targets.is_empty() {
            return Err(PlanError::NoTargets { refusals });
        }
        targets.sort_by(|a, b| {
            a.measures.end().total_cmp(&b.measures.end()).then_with(|| a.stream.cmp(&b.stream))
        });
        let plan = ForecastPlan { origin, horizon, parameters, targets, sources, refusals };
        plan.check();
        Ok(plan)
    }

    /// The invariants of §3.3, which the construction above establishes:
    /// each target ends after the origin and within the horizon; one stream's
    /// targets increase strictly and do not overlap. A failure is an internal
    /// error, not a user's.
    fn check(&self) {
        let reach = self.origin + self.horizon + TIME_TIE_EPS;
        for t in &self.targets {
            assert!(t.measures.end() > self.origin + TIME_TIE_EPS && t.measures.end() <= reach,
                "internal error: target {t:?} lies outside (origin {}, origin + horizon]", self.origin);
        }
        for (stream, _) in &self.sources {
            let mine: Vec<&Target> = self.targets_of(stream).collect();
            for w in mine.windows(2) {
                assert!(w[1].measures.start() >= w[0].measures.end() - TIME_TIE_EPS
                    && w[1].measures.end() > w[0].measures.end(),
                    "internal error: targets of '{stream}' overlap or do not increase: {w:?}");
            }
        }
    }

    /// The forecast origin: the training extent (D57).
    pub(crate) fn origin(&self) -> f64 {
        self.origin
    }

    /// The reach, a duration in model time units.
    pub(crate) fn horizon(&self) -> f64 {
        self.horizon
    }

    pub(crate) fn parameters(&self) -> &TrainingPartition {
        &self.parameters
    }

    /// Every target, sorted by (end, stream).
    pub(crate) fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// `stream`'s targets, in increasing order.
    pub(crate) fn targets_of<'a>(&'a self, stream: &'a str) -> impl Iterator<Item = &'a Target> + 'a {
        self.targets.iter().filter(move |t| t.stream == stream)
    }

    pub(crate) fn sources(&self) -> &[(String, TargetSource)] {
        &self.sources
    }

    pub(crate) fn refusals(&self) -> &[StreamRefusal] {
        &self.refusals
    }

    /// The latest end among the targets: how far the run must go (D49).
    pub(crate) fn run_end(&self) -> f64 {
        self.targets.iter().map(|t| t.measures.end()).fold(f64::NEG_INFINITY, f64::max)
    }
}

/// One stream's targets in `(origin, origin + horizon]`, and their source.
fn stream_targets(
    s: &PlanStream,
    origin: f64,
    horizon: f64,
) -> Result<(TargetSource, Vec<Target>), StreamRefusal> {
    let stream = s.obs.name.clone();
    let columns = crate::pfilter::stream_aux_columns(s.obs);
    if !columns.is_empty() {
        return Err(StreamRefusal::ReadsDataColumn { stream, columns });
    }
    let reach = origin + horizon + TIME_TIE_EPS;
    let after_origin = |end: f64| end > origin + TIME_TIE_EPS;
    let kind = s.obs.projection.temporal_kind();
    let (source, measures): (TargetSource, Vec<Measures>) = match (kind, &s.obs.covers) {
        (TemporalKind::Interval, Some(Covers::From { span, .. } | Covers::Until { span, .. })) => {
            let span = *span;
            // The phase: any row of a uniform stream fixes it, because its
            // periods tile without gaps; the last one is nearest the origin.
            let Some(anchor) = s.training.last().or(s.held_out.first().map(|h| &h.row)) else {
                return Err(StreamRefusal::NoBoundRows { stream });
            };
            let (a_start, a_end) = (anchor.measures.start(), anchor.measures.end());
            // The first whole step past the anchor whose period ends after the
            // origin; each period is computed from the anchor, not accumulated,
            // so no drift builds up over a long reach.
            let mut k = ((origin - a_end) / span).floor().max(0.0) as i64;
            let mut out = Vec::new();
            loop {
                let end = a_end + (k as f64) * span;
                if end > reach {
                    break;
                }
                if after_origin(end) {
                    let start = a_start + (k as f64) * span;
                    out.push(Measures::Period(period(start, end)));
                }
                k += 1;
            }
            (TargetSource::DeclaredTiling { span }, out)
        }
        (TemporalKind::Interval, _) => {
            let Some(times) = emit_times(s.obs, reach) else {
                return Err(StreamRefusal::NoRhythm { stream, kind });
            };
            let out = times.windows(2)
                .filter(|w| after_origin(w[1]) && w[1] <= reach)
                .map(|w| Measures::Period(period(w[0], w[1])))
                .collect();
            (TargetSource::EmitSchedule, out)
        }
        (TemporalKind::Instant, _) => {
            let Some(times) = emit_times(s.obs, reach) else {
                return Err(StreamRefusal::NoRhythm { stream, kind });
            };
            let out = times.into_iter()
                .filter(|&t| after_origin(t) && t <= reach)
                .map(Measures::Instant)
                .collect();
            (TargetSource::EmitSchedule, out)
        }
    };
    // A declared rhythm that lands on a training row means the rhythm and the
    // rows disagree: refuse rather than predict a period already scored.
    for m in &measures {
        if let Some(row) = s.training.iter().find(|r| overlaps(m, &r.measures)) {
            return Err(StreamRefusal::OverlapsObserved {
                stream, start: m.start(), end: m.end(), row: row.position,
            });
        }
    }
    let targets = measures.into_iter().enumerate()
        .map(|(h, m)| Target {
            stream: stream.clone(),
            observed: s.held_out.iter()
                .find(|h| same_span(&h.row.measures, &m))
                .map(|h| HeldOutRow { file: h.file, position: h.row.position }),
            measures: m,
            horizon: h as i64,
            lead: m.end() - origin,
        })
        .collect();
    Ok((source, targets))
}

/// The emit schedule's times up to `reach`, or `None` when the stream declares
/// none. A regular schedule continues past its baked `end`, which the runtime
/// ignores (it is the model horizon at compile time, gh#561).
fn emit_times(obs: &ObservationModel, reach: f64) -> Option<Vec<f64>> {
    match obs.emit_schedule.as_ref()? {
        ObservationSchedule::Regular(r) if r.step > 0.0 => {
            let mut out = Vec::new();
            let mut k = 0_u64;
            loop {
                let t = r.start + (k as f64) * r.step;
                if t > reach {
                    break;
                }
                out.push(t);
                k += 1;
            }
            Some(out)
        }
        ObservationSchedule::Regular(_) => None,
        ObservationSchedule::AtTimes(ts) => Some(ts.clone()),
    }
}

fn period(start: f64, end: f64) -> Period {
    Period::new(start, end).expect("a target period has a positive width by construction")
}

fn overlaps(a: &Measures, b: &Measures) -> bool {
    match (a, b) {
        (Measures::Instant(x), Measures::Instant(y)) => (x - y).abs() <= TIME_TIE_EPS,
        _ => a.start() < b.end() - TIME_TIE_EPS && b.start() < a.end() - TIME_TIE_EPS,
    }
}

fn same_span(a: &Measures, b: &Measures) -> bool {
    (a.start() - b.start()).abs() <= TIME_TIE_EPS && (a.end() - b.end()).abs() <= TIME_TIE_EPS
}

/// Parse the forecast's reach, `--horizon DURATION`: `N UNIT` with a plain unit
/// word (days, weeks, months, years; months and years as fixed spans), or a
/// bare number in model time units.
pub(crate) fn parse_reach(raw: &str, time_unit: &str) -> Result<f64, String> {
    let s = raw.trim();
    let bad = |why: &str| format!("--horizon \"{raw}\": {why} Write a duration such as \"4 weeks\" or \"28 days\".");
    if let Ok(v) = s.parse::<f64>() {
        if !(v.is_finite() && v > 0.0) {
            return Err(bad("the reach must be positive."));
        }
        return Ok(v);
    }
    let mut it = s.split_whitespace();
    let (Some(n), Some(unit), None) = (it.next(), it.next(), it.next()) else {
        return Err(bad("expected a number and a unit."));
    };
    let n: f64 = n.parse().map_err(|_| bad(&format!("'{n}' is not a number.")))?;
    if !(n.is_finite() && n > 0.0) {
        return Err(bad("the reach must be positive."));
    }
    let unit = super::runner::canonical_duration_unit(unit);
    let days = ir::caltime::days_per_unit(&unit).map_err(|_| bad(&format!("unknown unit '{unit}'.")))?;
    let model_days = ir::caltime::days_per_unit(time_unit)
        .map_err(|e| format!("--horizon: the model's time_unit: {e}"))?;
    Ok(n * days / model_days)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caltime_load::TemporalCell;
    use ir::observation::*;
    use std::collections::BTreeMap;

    fn obs(name: &str, covers: Option<Covers>, projection: Projection,
           emit: Option<ObservationSchedule>) -> ObservationModel {
        ObservationModel {
            name: name.into(),
            source: name.into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: name.into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: name.into(),
            emit_schedule: emit,
            stratum: vec![],
            covers,
            projection,
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    fn row(position: usize, start: f64, end: f64) -> ObservedRow {
        ObservedRow {
            position,
            label: end,
            measures: Measures::Period(Period::new(start, end).unwrap()),
            temporal: vec![TemporalCell::Number(end)],
            cell: Some(sim::inference::ObsCell::Scalar(1.0)),
            aux: vec![],
        }
    }

    fn instant_row(position: usize, t: f64) -> ObservedRow {
        ObservedRow {
            position,
            label: t,
            measures: Measures::Instant(t),
            temporal: vec![TemporalCell::Number(t)],
            cell: Some(sim::inference::ObsCell::Scalar(1.0)),
            aux: vec![],
        }
    }

    fn partition(rows: &BTreeMap<String, Vec<ObservedRow>>) -> TrainingPartition {
        TrainingPartition::whole(rows, super::super::training_partition::WholeFiles::NoHoldout).unwrap()
    }

    fn weekly() -> ObservationModel {
        obs("weekly", Some(Covers::Until { offset: 0.0, span: 7.0 }),
            Projection::CumulativeFlow("infection".into()), None)
    }

    fn daily() -> ObservationModel {
        obs("daily", Some(Covers::Until { offset: 0.0, span: 1.0 }),
            Projection::CumulativeFlow("infection".into()), None)
    }

    /// The spec's Figure 3: daily through day 24, weekly through day 21. The
    /// origin is 24; the weekly week [21, 28) is under way and is horizon 0;
    /// the daily days are horizons 0, 1, 2, …
    #[test]
    fn the_week_under_way_is_horizon_zero_and_each_stream_counts_its_own_periods() {
        let d: Vec<ObservedRow> = (0..24).map(|i| row(i, i as f64, i as f64 + 1.0)).collect();
        let w: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("daily".to_string(), d.clone()), ("weekly".to_string(), w.clone())]);
        let (dobs, wobs) = (daily(), weekly());
        let plan = ForecastPlan::new(partition(&rows), 7.0, &[
            PlanStream { obs: &dobs, training: &d, held_out: &[] },
            PlanStream { obs: &wobs, training: &w, held_out: &[] },
        ]).unwrap();
        assert_eq!(plan.origin(), 24.0);
        let weekly: Vec<&Target> = plan.targets_of("weekly").collect();
        assert_eq!(weekly.len(), 1, "{weekly:?}");
        assert_eq!((weekly[0].measures.start(), weekly[0].measures.end()), (21.0, 28.0));
        assert_eq!((weekly[0].horizon, weekly[0].lead), (0, 4.0));
        let daily: Vec<(f64, f64, i64)> = plan.targets_of("daily")
            .map(|t| (t.measures.start(), t.measures.end(), t.horizon)).collect();
        assert_eq!(daily, (0..7).map(|h| (24.0 + h as f64, 25.0 + h as f64, h)).collect::<Vec<_>>());
        assert_eq!(plan.run_end(), 31.0);
    }

    /// `--horizon "4 weeks"` from a week boundary: weekly horizons 0–3, daily
    /// horizons 0–27.
    #[test]
    fn four_weeks_from_a_week_boundary_gives_weekly_0_to_3_and_daily_0_to_27() {
        let d: Vec<ObservedRow> = (0..21).map(|i| row(i, i as f64, i as f64 + 1.0)).collect();
        let w: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("daily".to_string(), d.clone()), ("weekly".to_string(), w.clone())]);
        let (dobs, wobs) = (daily(), weekly());
        let reach = parse_reach("4 weeks", "days").unwrap();
        let plan = ForecastPlan::new(partition(&rows), reach, &[
            PlanStream { obs: &dobs, training: &d, held_out: &[] },
            PlanStream { obs: &wobs, training: &w, held_out: &[] },
        ]).unwrap();
        let wh: Vec<i64> = plan.targets_of("weekly").map(|t| t.horizon).collect();
        let dh: Vec<i64> = plan.targets_of("daily").map(|t| t.horizon).collect();
        assert_eq!(wh, vec![0, 1, 2, 3]);
        assert_eq!(dh, (0..28).collect::<Vec<_>>());
        let leads: Vec<f64> = plan.targets_of("weekly").map(|t| t.lead).collect();
        assert_eq!(leads, vec![7.0, 14.0, 21.0, 28.0]);
    }

    /// A `closing_at(time, 7 'days)` stream whose last training row is NA: the
    /// tail continues by the declared 7 days, never by the most common gap
    /// between rows with values.
    #[test]
    fn the_tail_continues_by_the_declared_width_not_the_modal_gap() {
        let mut w: Vec<ObservedRow> = (0..5).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        w[3].cell = None;
        let rows = BTreeMap::from([("weekly".to_string(), w.clone())]);
        let wobs = weekly();
        let plan = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &w, held_out: &[] },
        ]).unwrap();
        let spans: Vec<(f64, f64)> = plan.targets().iter().map(|t| (t.measures.start(), t.measures.end())).collect();
        assert_eq!(spans, vec![(35.0, 42.0), (42.0, 49.0)]);
    }

    /// Window columns with no `emit_schedule` declare no rhythm (D42); the
    /// stream is refused and, with no other stream, so is the plan.
    #[test]
    fn a_window_column_stream_without_an_emit_schedule_is_refused() {
        let w: Vec<ObservedRow> = vec![row(0, 0.0, 30.0), row(1, 30.0, 61.0)];
        let rows = BTreeMap::from([("monthly".to_string(), w.clone())]);
        let o = obs("monthly", Some(Covers::WindowColumns),
            Projection::CumulativeFlow("infection".into()), None);
        let err = ForecastPlan::new(partition(&rows), 90.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[] },
        ]).unwrap_err();
        let PlanError::NoTargets { refusals } = &err else { panic!("{err:?}") };
        assert!(matches!(refusals[..], [StreamRefusal::NoRhythm { .. }]), "{refusals:?}");
        assert!(err.to_string().contains("emit_schedule"), "{err}");
    }

    /// Window columns with an `emit_schedule`: the periods between consecutive
    /// schedule times.
    #[test]
    fn a_window_column_stream_takes_its_targets_from_the_emit_schedule() {
        let w: Vec<ObservedRow> = vec![row(0, 0.0, 30.0), row(1, 30.0, 60.0)];
        let rows = BTreeMap::from([("monthly".to_string(), w.clone())]);
        let o = obs("monthly", Some(Covers::WindowColumns),
            Projection::CumulativeFlow("infection".into()),
            Some(ObservationSchedule::AtTimes(vec![0.0, 30.0, 60.0, 91.0, 120.0])));
        let plan = ForecastPlan::new(partition(&rows), 61.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[] },
        ]).unwrap();
        let spans: Vec<(f64, f64)> = plan.targets().iter().map(|t| (t.measures.start(), t.measures.end())).collect();
        assert_eq!(spans, vec![(60.0, 91.0), (91.0, 120.0)]);
        assert_eq!(plan.sources()[0].1, TargetSource::EmitSchedule);
    }

    /// An emit schedule that lands a target on a training row is refused.
    #[test]
    fn an_emit_schedule_that_overlaps_a_training_row_is_refused() {
        let w: Vec<ObservedRow> = vec![row(0, 0.0, 30.0), row(1, 30.0, 60.0)];
        let rows = BTreeMap::from([("monthly".to_string(), w.clone())]);
        let o = obs("monthly", Some(Covers::WindowColumns),
            Projection::CumulativeFlow("infection".into()),
            Some(ObservationSchedule::AtTimes(vec![0.0, 45.0, 75.0])));
        let err = ForecastPlan::new(partition(&rows), 61.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[] },
        ]).unwrap_err();
        let PlanError::NoTargets { refusals } = &err else { panic!("{err:?}") };
        assert!(matches!(refusals[..], [StreamRefusal::OverlapsObserved { row: 1, .. }]), "{refusals:?}");
    }

    /// A state read takes the schedule's instants after the origin, or is
    /// refused without one.
    #[test]
    fn an_instant_stream_takes_the_emit_schedules_instants_or_is_refused() {
        let p: Vec<ObservedRow> = (0..3).map(|i| instant_row(i, 7.0 * i as f64)).collect();
        let rows = BTreeMap::from([("prev".to_string(), p.clone())]);
        let none = obs("prev", None, Projection::CurrentPop("I".into()), None);
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &none, training: &p, held_out: &[] },
        ]).unwrap_err();
        assert!(matches!(&err, PlanError::NoTargets { refusals } if matches!(refusals[..], [StreamRefusal::NoRhythm { kind: TemporalKind::Instant, .. }])), "{err:?}");
        let every = obs("prev", None, Projection::CurrentPop("I".into()),
            Some(ObservationSchedule::Regular(RegularSchedule { start: 0.0, step: 7.0, end: 14.0 })));
        let plan = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &every, training: &p, held_out: &[] },
        ]).unwrap();
        let at: Vec<(f64, i64)> = plan.targets().iter().map(|t| (t.measures.end(), t.horizon)).collect();
        assert_eq!(at, vec![(21.0, 0), (28.0, 1)], "the schedule continues past its baked end");
    }

    /// A likelihood that reads a data column (a binomial's `n`) is refused a
    /// forecast (D52), naming the column.
    #[test]
    fn a_stream_whose_likelihood_reads_a_data_column_is_refused() {
        let mut o = weekly();
        o.likelihood = Likelihood::Binomial(BinomialLikelihood {
            n: ir::expr::Expr::ObsColumnRef(ir::expr::ObsColumnRefExpr { obs_column_ref: "tested".into() }),
            p: ir::Diffable::new(ir::expr::Expr::const_(0.5)),
        });
        let w: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), w.clone())]);
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[] },
        ]).unwrap_err();
        assert!(err.to_string().contains("[tested]"), "{err}");
    }

    /// A held-out row measuring a target's period gives the target its file
    /// and its position within that file.
    #[test]
    fn a_held_out_row_names_its_target_by_file_and_position() {
        let all: Vec<ObservedRow> = (0..5).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let (train, rest) = all.split_at(3);
        let rows = BTreeMap::from([("weekly".to_string(), train.to_vec())]);
        let wobs = weekly();
        let held: Vec<HeldOut> = rest.iter()
            .map(|r| HeldOut { file: RowFile::Training, row: r.clone() })
            .collect();
        let plan = ForecastPlan::new(partition(&rows), 21.0, &[
            PlanStream { obs: &wobs, training: train, held_out: &held },
        ]).unwrap();
        let obs: Vec<Option<HeldOutRow>> = plan.targets().iter().map(|t| t.observed).collect();
        let at = |position| Some(HeldOutRow { file: RowFile::Training, position });
        assert_eq!(obs, vec![at(3), at(4), None]);
    }

    #[test]
    fn the_reach_parses_durations_in_the_models_unit() {
        assert_eq!(parse_reach("4 weeks", "days").unwrap(), 28.0);
        assert_eq!(parse_reach("28", "days").unwrap(), 28.0);
        assert_eq!(parse_reach("14 days", "weeks").unwrap(), 2.0);
        assert!(parse_reach("0 days", "days").is_err());
        assert!(parse_reach("four weeks", "days").is_err());
    }
}
