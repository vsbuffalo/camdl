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
    /// The design values the stream's likelihood reads at this target (D81),
    /// sorted by column; empty for a stream that reads none. They determine
    /// the prediction, so they are part of its identity.
    pub design: Vec<(String, f64)>,
}

/// Where a design-reading stream's design values for its targets come from
/// (D81).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DesignMode {
    /// Supplied at the origin by a design file: a prospective forecast.
    Prospective,
    /// The held-out rows' own design values, realised after the origin: an
    /// ex-post evaluation, a conditional score using realised future design,
    /// never an unconditional forecast.
    ExPost,
}

impl DesignMode {
    /// How a manifest and a message label the mode.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            DesignMode::Prospective => "a prospective forecast: the design values were supplied at \
                the forecast origin by a design file",
            DesignMode::ExPost => "a conditional score using realised future design: the design \
                values are the held-out rows' own, realised after the forecast origin, so this is \
                an ex-post evaluation and not an unconditional forecast",
        }
    }
}

/// What one stream's design values come from (D81).
#[derive(Debug, Clone, Copy)]
pub(crate) enum DesignInput<'a> {
    /// No design was supplied: a stream that reads one is refused.
    NotSupplied,
    /// The rows of a design file (`--design`), in file order.
    Prospective(&'a [ObservedRow]),
    /// The stream's held-out rows (`--realised-design`).
    ExPost,
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
    /// A window-column stream with no `emit_schedule`, scored: the periods of
    /// its held-out rows within the reach (D80). Nothing is extrapolated, and
    /// a target's ordinal counts only the held-out rows, so it is not a
    /// horizon comparable with a tiled stream's; compare such targets by lead.
    HeldOutRows,
}

impl TargetSource {
    pub(crate) fn describe(&self) -> String {
        match self {
            TargetSource::DeclaredTiling { span } => format!("declared_tiling(span = {span})"),
            TargetSource::EmitSchedule => "emit_schedule".to_string(),
            TargetSource::HeldOutRows => "held_out_rows (each target's horizon is its position \
                among the held-out rows, not a count of declared periods; compare by lead)"
                .to_string(),
        }
    }

    /// Whether the stream declares a cadence of its own (§4.3, 1 and 2), so
    /// held-out rows only select its targets.
    fn is_declared_cadence(&self) -> bool {
        match self {
            TargetSource::DeclaredTiling { .. } | TargetSource::EmitSchedule => true,
            TargetSource::HeldOutRows => false,
        }
    }
}

impl HeldOutRow {
    /// How a message names the row.
    pub(crate) fn describe(&self) -> String {
        format!("held-out row {} of the {} file", self.position, self.file.as_str())
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
    /// Where the stream's design values come from, when its likelihood reads
    /// any (D81).
    pub design: DesignInput<'a>,
}

/// Why a stream has no targets (§4.3, D42, D81).
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
    /// A held-out row within the reach is not one of the targets the stream's
    /// declared cadence gives (D80): refused rather than skipped, since a
    /// skipped row would silently leave a period out of the score.
    OffCadenceHeldOut { stream: String, start: f64, end: f64, row: HeldOutRow, source: TargetSource },
    /// The held-out rows within the reach are out of order, repeat a period,
    /// overlap each other or overlap the training rows (D80).
    InvalidHeldOut { stream: String, row: HeldOutRow, why: String },
    /// The periods the stream's data imply and the periods of its
    /// `emit_schedule` disagree over the same reporting interval (D82). There
    /// is no precedence rule, so neither is followed.
    ScheduleConflict {
        stream: String,
        data: (f64, f64),
        /// Where `data` comes from: a training row, a held-out row, or the
        /// declared tiling continued past the rows.
        data_source: String,
        schedule: (f64, f64),
    },
    /// A target of a design-reading stream has no value of a design column
    /// in the supplied design (D81): refused, never filled in.
    MissingDesign { stream: String, column: String, start: f64, end: f64, mode: DesignMode, why: String },
    /// A row of a `--design` file within the reach names no target (D81).
    DesignOffTarget { stream: String, start: f64, end: f64, row: usize },
}

impl StreamRefusal {
    /// The refused stream's leaf name.
    pub(crate) fn stream(&self) -> &str {
        match self {
            StreamRefusal::NoRhythm { stream, .. }
            | StreamRefusal::NoBoundRows { stream }
            | StreamRefusal::ReadsDataColumn { stream, .. }
            | StreamRefusal::OverlapsObserved { stream, .. }
            | StreamRefusal::OffCadenceHeldOut { stream, .. }
            | StreamRefusal::InvalidHeldOut { stream, .. }
            | StreamRefusal::ScheduleConflict { stream, .. }
            | StreamRefusal::MissingDesign { stream, .. }
            | StreamRefusal::DesignOffTarget { stream, .. } => stream,
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
                 column(s) [{}], which have no value for a target past the data unless one \
                 is supplied. Pass --design SOURCE=FILE, a file in the stream's own format \
                 giving the column(s) for each target (a prospective forecast), or \
                 --realised-design to read them from the held-out rows (a conditional score \
                 using realised future design, not an unconditional forecast).",
                columns.join(", ")),
            StreamRefusal::MissingDesign { stream, column, start, end, mode, why } => write!(f,
                "stream '{stream}' is refused a forecast: its target [{start}, {end}) has no \
                 value of the design column '{column}' ({}: {why}). A missing design value is \
                 not filled in.",
                match mode {
                    DesignMode::Prospective => "--design",
                    DesignMode::ExPost => "--realised-design",
                }),
            StreamRefusal::DesignOffTarget { stream, start, end, row } => write!(f,
                "stream '{stream}' is refused a forecast: row {row} of its --design file \
                 covers [{start}, {end}), which is not one of its targets within the reach. \
                 A design row names the target it is for; one that names none is not \
                 skipped."),
            StreamRefusal::OverlapsObserved { stream, start, end, row } => write!(f,
                "stream '{stream}' is refused a forecast: its declared rhythm puts a \
                 target at [{start}, {end}), which overlaps its training row {row}. The \
                 rhythm and the rows disagree about where the stream's periods fall."),
            StreamRefusal::OffCadenceHeldOut { stream, start, end, row, source } => write!(f,
                "stream '{stream}' is refused a forecast: its {} covers [{start}, {end}), \
                 which is not one of the periods its declared rhythm ({}) gives within the \
                 reach. A held-out row only selects which of those periods is scored; one \
                 off the rhythm is not skipped, because the score would then leave it out \
                 without saying so. Correct the row, or the declaration.",
                row.describe(), source.describe()),
            StreamRefusal::InvalidHeldOut { stream, row, why } => write!(f,
                "stream '{stream}' is refused a forecast: its {} {why}.", row.describe()),
            StreamRefusal::ScheduleConflict { stream, data, data_source, schedule } => write!(f,
                "stream '{stream}' is refused a forecast: its data imply the period [{}, {}) \
                 ({data_source}), but its emit_schedule gives [{}, {}) over the same \
                 reporting interval. Neither takes precedence; make the declarations agree \
                 (or remove the emit_schedule). A deliberately different prospective \
                 schedule would need its own declaration, which camdl does not have.",
                data.0, data.1, schedule.0, schedule.1),
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
    /// Where the design values of the design-reading streams with targets
    /// came from (D81); `None` when no such stream has targets.
    design_mode: Option<DesignMode>,
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
        let mut design_modes: Vec<DesignMode> = Vec::new();
        for s in streams {
            match stream_targets(s, origin, horizon) {
                Ok((source, ts)) => {
                    if !ts.is_empty() {
                        if ts.iter().any(|t| !t.design.is_empty()) {
                            design_modes.push(match s.design {
                                DesignInput::Prospective(_) => DesignMode::Prospective,
                                DesignInput::ExPost => DesignMode::ExPost,
                                DesignInput::NotSupplied => unreachable!(
                                    "a target carries design values only when a design was supplied"),
                            });
                        }
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
        design_modes.dedup();
        let design_mode = match design_modes[..] {
            [] => None,
            [m] => Some(m),
            _ => unreachable!("one forecast reads its design one way: the command takes one mode"),
        };
        let plan = ForecastPlan { origin, horizon, parameters, targets, sources, refusals, design_mode };
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

    /// Where the design values came from, when any target carries some (D81).
    pub(crate) fn design_mode(&self) -> Option<DesignMode> {
        self.design_mode
    }

    /// The SHA-256 of the target set, in the plan's order: each target's
    /// stream, whether it is a period or an instant, its start and end, and
    /// its design values where it reads any. These are inputs to the
    /// predictions (D80: under held-out-row targets the set follows the
    /// held-out periods; D81: the design values), so they key the forecast's
    /// address. Held-out outcomes do not enter it.
    pub(crate) fn targets_digest(&self) -> String {
        let mut h = String::new();
        for t in &self.targets {
            let kind = match t.measures {
                Measures::Period(_) => "period",
                Measures::Instant(_) => "instant",
            };
            h += &format!("{}\t{kind}\t{:016x}\t{:016x}", t.stream, t.measures.start().to_bits(), t.measures.end().to_bits());
            for (c, v) in &t.design {
                h += &format!("\t{c}={:016x}", v.to_bits());
            }
            h += "\n";
        }
        crate::hashing::sha256_hex(h.as_bytes())
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
    // D81 (superseding D52): a stream whose likelihood reads design columns
    // is forecast only when a design supplies them.
    let columns = crate::pfilter::stream_aux_columns(s.obs);
    if !columns.is_empty() && matches!(s.design, DesignInput::NotSupplied) {
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
        // §4.3 (3): a window-column stream with no emit schedule, scored,
        // takes its held-out rows' periods as targets; with no held-out row it
        // declares no rhythm and falls to the refusal below.
        (TemporalKind::Interval, Some(Covers::WindowColumns))
            if s.obs.emit_schedule.is_none() && !s.held_out.is_empty() =>
        {
            let in_reach: Vec<&HeldOut> = s.held_out.iter()
                .filter(|h| after_origin(h.row.measures.end()) && h.row.measures.end() <= reach)
                .collect();
            validate_held_out_periods(&stream, s.training, &in_reach)?;
            (TargetSource::HeldOutRows, in_reach.iter().map(|h| h.row.measures).collect())
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
    // D82: where the data and the emit schedule both say what a reporting
    // period is, they must say the same thing.
    if kind == TemporalKind::Interval {
        check_schedule_agrees(s, &stream, source, &measures, reach)?;
    }
    // A declared rhythm that lands on a training row means the rhythm and the
    // rows disagree: refuse rather than predict a period already scored.
    for m in &measures {
        if let Some(row) = s.training.iter().find(|r| overlaps(m, &r.measures)) {
            return Err(StreamRefusal::OverlapsObserved {
                stream, start: m.start(), end: m.end(), row: row.position,
            });
        }
    }
    // D80: a held-out row within the reach names exactly one target. Under a
    // declared cadence it only selects which target is scored, so one that is
    // not a target is refused, not skipped; two naming one target are refused
    // rather than resolved by file order.
    let mut observed: Vec<Option<HeldOutRow>> = vec![None; measures.len()];
    for h in s.held_out.iter().filter(|h| after_origin(h.row.measures.end()) && h.row.measures.end() <= reach) {
        let here = HeldOutRow { file: h.file, position: h.row.position };
        match measures.iter().position(|m| same_span(&h.row.measures, m)) {
            Some(k) => match observed[k] {
                None => observed[k] = Some(here),
                Some(first) => return Err(StreamRefusal::InvalidHeldOut {
                    stream, row: here,
                    why: format!("repeats the period [{}, {}) of its {}; each target is scored \
                        against one row", h.row.measures.start(), h.row.measures.end(), first.describe()),
                }),
            },
            None if source.is_declared_cadence() => return Err(StreamRefusal::OffCadenceHeldOut {
                stream, start: h.row.measures.start(), end: h.row.measures.end(), row: here, source,
            }),
            None => unreachable!("held-out rows within the reach are this source's targets"),
        }
    }
    let design = target_design(s, &stream, &columns, &measures, &observed, origin, reach)?;
    let targets = measures.into_iter().zip(observed).zip(design).enumerate()
        .map(|(h, ((m, observed), design))| Target {
            stream: stream.clone(),
            observed,
            measures: m,
            horizon: h as i64,
            lead: m.end() - origin,
            design,
        })
        .collect();
    Ok((source, targets))
}

/// D81: each target's design values, `columns` of the stream's likelihood,
/// from where `s.design` says. A target with no value of a column is
/// refused, never filled in. A prospective design row within the reach that
/// names no target is refused, and so are two naming one. Outcomes never
/// reach the result: an ex-post target reads its held-out row's design as
/// read ([`ObservedRow::design`]), whatever its outcome.
fn target_design(
    s: &PlanStream,
    stream: &str,
    columns: &[String],
    targets: &[Measures],
    observed: &[Option<HeldOutRow>],
    origin: f64,
    reach: f64,
) -> Result<Vec<Vec<(String, f64)>>, StreamRefusal> {
    if columns.is_empty() {
        return Ok(vec![Vec::new(); targets.len()]);
    }
    let pick = |mode: DesignMode, m: &Measures, row: Option<&ObservedRow>, absent: &str|
        -> Result<Vec<(String, f64)>, StreamRefusal>
    {
        let missing = |column: &str, why: String| StreamRefusal::MissingDesign {
            stream: stream.to_string(), column: column.to_string(), start: m.start(), end: m.end(),
            mode, why,
        };
        let Some(row) = row else { return Err(missing(&columns[0], absent.to_string())) };
        let mut out = Vec::with_capacity(columns.len());
        for c in columns {
            match row.design.iter().find(|(k, _)| k == c) {
                Some(&(_, v)) if v.is_finite() => out.push((c.clone(), v)),
                Some(&(_, v)) => return Err(missing(c, format!("its value is {v}"))),
                None => return Err(missing(c, "the row's cell is NA".into())),
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    };
    match s.design {
        DesignInput::NotSupplied => Err(StreamRefusal::ReadsDataColumn {
            stream: stream.to_string(), columns: columns.to_vec(),
        }),
        DesignInput::Prospective(rows) => {
            let mut by_target: Vec<Option<&ObservedRow>> = vec![None; targets.len()];
            for r in rows.iter().filter(|r| r.measures.end() > origin + TIME_TIE_EPS && r.measures.end() <= reach) {
                match targets.iter().position(|m| same_span(&r.measures, m)) {
                    None => return Err(StreamRefusal::DesignOffTarget {
                        stream: stream.to_string(), start: r.measures.start(), end: r.measures.end(),
                        row: r.position,
                    }),
                    Some(k) if by_target[k].is_some() => return Err(StreamRefusal::MissingDesign {
                        stream: stream.to_string(), column: columns[0].clone(),
                        start: targets[k].start(), end: targets[k].end(), mode: DesignMode::Prospective,
                        why: format!("rows {} and {} of the design file both name it, so neither is used",
                            by_target[k].map(|p| p.position).unwrap_or_default(), r.position),
                    }),
                    Some(k) => by_target[k] = Some(r),
                }
            }
            targets.iter().zip(by_target)
                .map(|(m, row)| pick(DesignMode::Prospective, m, row, "the design file has no row for it"))
                .collect()
        }
        DesignInput::ExPost => targets.iter().zip(observed)
            .map(|(m, at)| {
                let row = at.and_then(|at| s.held_out.iter()
                    .find(|h| h.file == at.file && h.row.position == at.position)
                    .map(|h| &h.row));
                pick(DesignMode::ExPost, m, row, "no held-out row measures it")
            })
            .collect(),
    }
}

/// D82: an interval stream's data-implied periods (its training rows, its
/// held-out rows and, for a declared tiling, the targets the tiling continues
/// to) against the periods between consecutive times of its `emit_schedule`.
/// A data period that overlaps a schedule period without being it is a
/// conflict. Only the reporting intervals this forecast reaches are judged,
/// the reach D80 uses: data periods ending at or before `reach` (a held-out
/// row past it is not this forecast's), against every schedule period that
/// overlaps them, one crossing the reach included. A stream with no emit
/// schedule, or a schedule with no period over a data period, has nothing to
/// disagree with.
fn check_schedule_agrees(
    s: &PlanStream,
    stream: &str,
    source: TargetSource,
    targets: &[Measures],
    reach: f64,
) -> Result<(), StreamRefusal> {
    let Some(mut times) = emit_times(s.obs, reach) else { return Ok(()) };
    // The schedule period that crosses the reach, so a data period ending at
    // or before it is compared with every period it overlaps.
    if let Some(ObservationSchedule::Regular(r)) = s.obs.emit_schedule.as_ref() {
        if let Some(&last) = times.last() {
            times.push(last + r.step);
        }
    }
    let schedule: Vec<Measures> = times.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| Measures::Period(period(w[0], w[1])))
        .collect();
    let tiled = match source {
        TargetSource::DeclaredTiling { span } => targets.iter()
            .map(|m| (*m, format!("the declared tiling of `covers` (span {span}) continued past the rows")))
            .collect(),
        TargetSource::EmitSchedule | TargetSource::HeldOutRows => Vec::new(),
    };
    let data = s.training.iter()
        .map(|r| (r.measures, format!("training row {}", r.position)))
        .chain(s.held_out.iter()
            .map(|h| (h.row.measures, HeldOutRow { file: h.file, position: h.row.position }.describe())))
        .chain(tiled)
        .filter(|(m, _)| m.end() <= reach);
    for (m, from) in data {
        if let Some(q) = schedule.iter().find(|q| overlaps(&m, q) && !same_span(&m, q)) {
            return Err(StreamRefusal::ScheduleConflict {
                stream: stream.to_string(),
                data: (m.start(), m.end()),
                data_source: from,
                schedule: (q.start(), q.end()),
            });
        }
    }
    Ok(())
}

/// D80: the held-out rows within the reach of a window-column stream with no
/// emit schedule, which become its targets, must be in order, distinct and
/// disjoint, and must not overlap the stream's training rows. Each is a
/// `Period`, so its endpoints are finite and ordered by construction.
fn validate_held_out_periods(
    stream: &str,
    training: &[ObservedRow],
    held: &[&HeldOut],
) -> Result<(), StreamRefusal> {
    let refuse = |h: &HeldOut, why: String| StreamRefusal::InvalidHeldOut {
        stream: stream.to_string(), row: HeldOutRow { file: h.file, position: h.row.position }, why,
    };
    let span = |m: &Measures| format!("[{}, {})", m.start(), m.end());
    for h in held {
        if let Some(r) = training.iter().find(|r| overlaps(&r.measures, &h.row.measures)) {
            return Err(refuse(h, format!(
                "covers {}, which overlaps its training row {} ({}); a held-out window must \
                 begin where the training rows end", span(&h.row.measures), r.position, span(&r.measures))));
        }
    }
    for w in held.windows(2) {
        let (a, b) = (&w[0].row.measures, &w[1].row.measures);
        let first = HeldOutRow { file: w[0].file, position: w[0].row.position };
        if same_span(a, b) {
            return Err(refuse(w[1], format!("repeats the period {} of its {}", span(a), first.describe())));
        }
        if b.end() < a.end() {
            return Err(refuse(w[1], format!(
                "covers {}, which ends before the {} before it ({}): the held-out rows are out \
                 of order", span(b), first.describe(), span(a))));
        }
        if overlaps(a, b) {
            return Err(refuse(w[1], format!(
                "covers {}, which overlaps {} ({})", span(b), first.describe(), span(a))));
        }
    }
    Ok(())
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
/// word (days, weeks, months, years), or a bare number in model time units.
/// Without an origin months and years are fixed spans; under a calendar
/// [origin] they are refused, since the reach is counted from the forecast
/// origin, a date known only at run time (decision F, as for `--to`).
pub(crate) fn parse_reach(raw: &str, time_unit: &str, origin: Option<&str>) -> Result<f64, String> {
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
    let unit_tok = unit;
    let unit = super::runner::canonical_duration_unit(unit);
    let days = ir::caltime::days_per_unit(&unit).map_err(|_| bad(&format!("unknown unit '{unit}'.")))?;
    if origin.is_some() && super::runner::is_calendar_unit(&unit) {
        return Err(super::runner::calendar_offset_refusal(
            &format!("--horizon \"{raw}\""), &format!("{n} {unit_tok}"),
            "the forecast origin (the last training observation)", n * days,
            |d| format!("{d} days")));
    }
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
            design: vec![],
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
            design: vec![],
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
            PlanStream { obs: &dobs, training: &d, held_out: &[], design: DesignInput::NotSupplied },
            PlanStream { obs: &wobs, training: &w, held_out: &[], design: DesignInput::NotSupplied },
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
        let reach = parse_reach("4 weeks", "days", None).unwrap();
        let plan = ForecastPlan::new(partition(&rows), reach, &[
            PlanStream { obs: &dobs, training: &d, held_out: &[], design: DesignInput::NotSupplied },
            PlanStream { obs: &wobs, training: &w, held_out: &[], design: DesignInput::NotSupplied },
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
            PlanStream { obs: &wobs, training: &w, held_out: &[], design: DesignInput::NotSupplied },
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
            PlanStream { obs: &o, training: &w, held_out: &[], design: DesignInput::NotSupplied },
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
            PlanStream { obs: &o, training: &w, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap();
        let spans: Vec<(f64, f64)> = plan.targets().iter().map(|t| (t.measures.start(), t.measures.end())).collect();
        assert_eq!(spans, vec![(60.0, 91.0), (91.0, 120.0)]);
        assert_eq!(plan.sources()[0].1, TargetSource::EmitSchedule);
    }

    /// An emit schedule whose periods cut across the training rows is
    /// refused, at the first row it disagrees with (D82): here [0, 30) against
    /// the schedule's [0, 45), before the target [45, 75) that overlaps row 1.
    #[test]
    fn an_emit_schedule_that_overlaps_a_training_row_is_refused() {
        let w: Vec<ObservedRow> = vec![row(0, 0.0, 30.0), row(1, 30.0, 60.0)];
        let rows = BTreeMap::from([("monthly".to_string(), w.clone())]);
        let o = obs("monthly", Some(Covers::WindowColumns),
            Projection::CumulativeFlow("infection".into()),
            Some(ObservationSchedule::AtTimes(vec![0.0, 45.0, 75.0])));
        let err = ForecastPlan::new(partition(&rows), 61.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        let PlanError::NoTargets { refusals } = &err else { panic!("{err:?}") };
        assert_eq!(refusals, &[StreamRefusal::ScheduleConflict {
            stream: "monthly".into(), data: (0.0, 30.0), data_source: "training row 0".into(),
            schedule: (0.0, 45.0),
        }]);
    }

    /// A state read takes the schedule's instants after the origin, or is
    /// refused without one.
    #[test]
    fn an_instant_stream_takes_the_emit_schedules_instants_or_is_refused() {
        let p: Vec<ObservedRow> = (0..3).map(|i| instant_row(i, 7.0 * i as f64)).collect();
        let rows = BTreeMap::from([("prev".to_string(), p.clone())]);
        let none = obs("prev", None, Projection::CurrentPop("I".into()), None);
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &none, training: &p, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        assert!(matches!(&err, PlanError::NoTargets { refusals } if matches!(refusals[..], [StreamRefusal::NoRhythm { kind: TemporalKind::Instant, .. }])), "{err:?}");
        let every = obs("prev", None, Projection::CurrentPop("I".into()),
            Some(ObservationSchedule::Regular(RegularSchedule { start: 0.0, step: 7.0, end: 14.0 })));
        let plan = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &every, training: &p, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap();
        let at: Vec<(f64, i64)> = plan.targets().iter().map(|t| (t.measures.end(), t.horizon)).collect();
        assert_eq!(at, vec![(21.0, 0), (28.0, 1)], "the schedule continues past its baked end");
    }

    /// A likelihood that reads a data column (a binomial's `n`) is refused a
    /// forecast when no design supplies it (D81), naming the column and the
    /// two ways to supply it.
    #[test]
    fn a_stream_whose_likelihood_reads_a_data_column_is_refused() {
        let o = tested_weekly();
        let w: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), w.clone())]);
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[tested]") && msg.contains("--design") && msg.contains("--realised-design"), "{msg}");
    }

    fn tested_weekly() -> ObservationModel {
        let mut o = weekly();
        o.likelihood = Likelihood::Binomial(BinomialLikelihood {
            n: ir::expr::Expr::ObsColumnRef(ir::expr::ObsColumnRefExpr { obs_column_ref: "tested".into() }),
            p: ir::Diffable::new(ir::expr::Expr::const_(0.5)),
        });
        o
    }

    /// A row carrying a value and a `tested` design value.
    fn tested_row(position: usize, start: f64, end: f64, y: Option<f64>, tested: Option<f64>) -> ObservedRow {
        let design: Vec<(String, f64)> = tested.map(|n| ("tested".to_string(), n)).into_iter().collect();
        ObservedRow {
            cell: y.map(sim::inference::ObsCell::Scalar),
            aux: if y.is_some() { design.clone() } else { vec![] },
            design,
            ..row(position, start, end)
        }
    }

    /// D81, prospective: each target reads its design row's values; a target
    /// with no row, a row whose value is NA, and a row that names no target
    /// are each refused.
    #[test]
    fn a_prospective_design_supplies_each_targets_design_values() {
        let o = tested_weekly();
        let w: Vec<ObservedRow> = (0..3).map(|i| tested_row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64, Some(2.0), Some(10.0))).collect();
        let rows = BTreeMap::from([("weekly".to_string(), w.clone())]);
        let plan_with = |design: &[ObservedRow]| ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &o, training: &w, held_out: &[], design: DesignInput::Prospective(design) },
        ]);
        let design = vec![tested_row(0, 21.0, 28.0, None, Some(40.0)), tested_row(1, 28.0, 35.0, None, Some(50.0))];
        let plan = plan_with(&design).unwrap();
        let got: Vec<Vec<(String, f64)>> = plan.targets().iter().map(|t| t.design.clone()).collect();
        assert_eq!(got, vec![vec![("tested".to_string(), 40.0)], vec![("tested".to_string(), 50.0)]]);
        assert_eq!(plan.design_mode(), Some(DesignMode::Prospective));
        let refused = |design: Vec<ObservedRow>, want: &str| {
            let err = plan_with(&design).unwrap_err().to_string();
            assert!(err.contains(want), "want {want:?}: {err}");
        };
        refused(vec![design[0].clone()], "[28, 35) has no value of the design column 'tested' (--design: the design file has no row for it)");
        refused(vec![design[0].clone(), tested_row(1, 28.0, 35.0, None, None)], "the row's cell is NA");
        refused(vec![design[0].clone(), tested_row(1, 28.0, 31.0, None, Some(5.0))], "row 1 of its --design file covers [28, 31)");
    }

    /// D81, ex post: each target reads its held-out row's design as read,
    /// whatever the row's outcome; a target with no held-out row is refused.
    /// The design digest follows the design values and not the outcomes.
    #[test]
    fn an_ex_post_design_reads_the_held_out_rows_and_never_their_outcomes() {
        let o = tested_weekly();
        let w: Vec<ObservedRow> = (0..3).map(|i| tested_row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64, Some(2.0), Some(10.0))).collect();
        let rows = BTreeMap::from([("weekly".to_string(), w.clone())]);
        let held = |y0: Option<f64>, y1: Option<f64>, n1: f64| vec![
            HeldOut { file: RowFile::Training, row: tested_row(3, 21.0, 28.0, y0, Some(40.0)) },
            HeldOut { file: RowFile::Training, row: tested_row(4, 28.0, 35.0, y1, Some(n1)) },
        ];
        let plan_of = |h: &[HeldOut], reach: f64| ForecastPlan::new(partition(&rows), reach, &[
            PlanStream { obs: &o, training: &w, held_out: h, design: DesignInput::ExPost },
        ]);
        let (a, b, na) = (held(Some(3.0), Some(4.0), 50.0), held(Some(9.0), Some(1.0), 50.0), held(None, Some(4.0), 50.0));
        let (pa, pb, pna) = (plan_of(&a, 14.0).unwrap(), plan_of(&b, 14.0).unwrap(), plan_of(&na, 14.0).unwrap());
        assert_eq!(pa.design_mode(), Some(DesignMode::ExPost));
        assert_eq!(pa.targets(), pb.targets(), "outcomes change no target or design value");
        assert_eq!(pa.targets(), pna.targets(), "an NA outcome keeps its row's design");
        assert_eq!(pa.targets_digest(), pb.targets_digest());
        let changed = plan_of(&held(Some(3.0), Some(4.0), 51.0), 14.0).unwrap();
        assert_ne!(pa.targets_digest(), changed.targets_digest(), "the design values key the prediction");
        let err = plan_of(&a, 21.0).unwrap_err().to_string();
        assert!(err.contains("[35, 42)") && err.contains("no held-out row measures it"), "{err}");
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
            PlanStream { obs: &wobs, training: train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap();
        let obs: Vec<Option<HeldOutRow>> = plan.targets().iter().map(|t| t.observed).collect();
        let at = |position| Some(HeldOutRow { file: RowFile::Training, position });
        assert_eq!(obs, vec![at(3), at(4), None]);
    }

    /// D82: a uniform weekly `covers` form whose rows close on days 7, 14,
    /// 21 and an `emit_schedule` of every 7 days from day 3 disagree over the
    /// same reporting interval. The forecast is refused, naming the stream,
    /// both periods and the source of each, rather than following the data.
    #[test]
    fn a_tiling_and_an_emit_schedule_that_disagree_are_refused() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let mut wobs = weekly();
        wobs.emit_schedule = Some(ObservationSchedule::Regular(RegularSchedule { start: 3.0, step: 7.0, end: 63.0 }));
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        let PlanError::NoTargets { refusals } = &err else { panic!("{err:?}") };
        assert!(matches!(&refusals[..], [StreamRefusal::ScheduleConflict { .. }]), "{refusals:?}");
        let msg = err.to_string();
        assert!(msg.contains("'weekly'") && msg.contains("[0, 7)") && msg.contains("training row 0")
            && msg.contains("[3, 10)") && msg.contains("emit_schedule"), "{msg}");
        // Agreeing declarations forecast as before.
        wobs.emit_schedule = Some(ObservationSchedule::Regular(RegularSchedule { start: 0.0, step: 7.0, end: 63.0 }));
        let plan = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap();
        assert_eq!(plan.targets().len(), 2);
    }

    /// D82 on held-out data and the tiling's continuation: the schedule
    /// agrees over the training rows but not past them.
    #[test]
    fn a_schedule_that_disagrees_only_past_the_training_rows_is_refused() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let mut wobs = weekly();
        wobs.emit_schedule = Some(ObservationSchedule::AtTimes(vec![0.0, 7.0, 14.0, 21.0, 30.0]));
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[21, 28)") && msg.contains("declared tiling") && msg.contains("[21, 30)"), "{msg}");
    }

    /// D82 judges the reporting intervals this forecast reaches, as D80
    /// does: a held-out row past the reach that cuts across the schedule
    /// does not refuse a forecast whose targets agree with it.
    #[test]
    fn a_schedule_conflict_past_the_reach_does_not_refuse_the_forecast() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let mut wobs = weekly();
        wobs.emit_schedule = Some(ObservationSchedule::Regular(RegularSchedule { start: 0.0, step: 7.0, end: 63.0 }));
        let held = vec![HeldOut { file: RowFile::Holdout, row: row(0, 50.0, 57.0) }];
        let plan = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap();
        assert_eq!(plan.targets().len(), 2);
        // Within the reach the same row is refused.
        let err = ForecastPlan::new(partition(&rows), 42.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap_err();
        assert!(matches!(&err, PlanError::NoTargets { refusals } if matches!(refusals[..], [StreamRefusal::ScheduleConflict { .. }])), "{err:?}");
    }

    /// D80: a held-out row of a stream with a declared cadence that is not
    /// one of the stream's targets is refused, never skipped: a weekly stream
    /// whose held-out file carries a three-day row inside the reach.
    #[test]
    fn a_held_out_row_off_the_declared_cadence_is_refused() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let wobs = weekly();
        let held = vec![
            HeldOut { file: RowFile::Holdout, row: row(0, 21.0, 28.0) },
            HeldOut { file: RowFile::Holdout, row: row(1, 28.0, 31.0) },
        ];
        let err = ForecastPlan::new(partition(&rows), 14.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap_err();
        let PlanError::NoTargets { refusals } = &err else { panic!("{err:?}") };
        assert!(matches!(&refusals[..], [StreamRefusal::OffCadenceHeldOut {
            row: HeldOutRow { file: RowFile::Holdout, position: 1 }, .. }]), "{refusals:?}");
        let msg = err.to_string();
        assert!(msg.contains("[28, 31)") && msg.contains("declared_tiling"), "{msg}");
        // A held-out row past the reach is not this forecast's to judge.
        let ok = ForecastPlan::new(partition(&rows), 7.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap();
        assert_eq!(ok.targets()[0].observed, Some(HeldOutRow { file: RowFile::Holdout, position: 0 }));
    }

    /// D80: a held-out row of a declared-cadence stream selects a target and
    /// does not renumber it: with the second week missing, the third keeps
    /// horizon 2.
    #[test]
    fn held_out_rows_select_targets_without_renumbering_them() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let wobs = weekly();
        let held = vec![
            HeldOut { file: RowFile::Training, row: row(3, 21.0, 28.0) },
            HeldOut { file: RowFile::Training, row: row(4, 35.0, 42.0) },
        ];
        let plan = ForecastPlan::new(partition(&rows), 21.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap();
        let got: Vec<(i64, f64, Option<usize>)> = plan.targets().iter()
            .map(|t| (t.horizon, t.lead, t.observed.map(|h| h.position))).collect();
        assert_eq!(got, vec![(0, 7.0, Some(3)), (1, 14.0, None), (2, 21.0, Some(4))]);
    }

    /// D80: two held-out rows naming one target are refused, not resolved by
    /// order.
    #[test]
    fn two_held_out_rows_for_one_target_are_refused() {
        let train: Vec<ObservedRow> = (0..3).map(|i| row(i, 7.0 * i as f64, 7.0 * (i + 1) as f64)).collect();
        let rows = BTreeMap::from([("weekly".to_string(), train.clone())]);
        let wobs = weekly();
        let held = vec![
            HeldOut { file: RowFile::Training, row: row(3, 21.0, 28.0) },
            HeldOut { file: RowFile::Holdout, row: row(0, 21.0, 28.0) },
        ];
        let err = ForecastPlan::new(partition(&rows), 7.0, &[
            PlanStream { obs: &wobs, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap_err();
        assert!(err.to_string().contains("held-out row 0 of the holdout file"), "{err}");
    }

    fn monthly_windows() -> ObservationModel {
        obs("monthly", Some(Covers::WindowColumns), Projection::CumulativeFlow("infection".into()), None)
    }

    /// D80: a window-column stream with no `emit_schedule` takes its
    /// held-out periods as targets when there are held-out rows; each carries
    /// its lead, and its ordinal is labelled as not a tiled horizon.
    #[test]
    fn a_window_column_stream_takes_its_held_out_periods_as_targets() {
        let train = vec![row(0, 0.0, 30.0), row(1, 30.0, 61.0)];
        let rows = BTreeMap::from([("monthly".to_string(), train.clone())]);
        let o = monthly_windows();
        // A gap between the two held-out windows, and a third past the reach.
        let held = vec![
            HeldOut { file: RowFile::Training, row: row(2, 61.0, 92.0) },
            HeldOut { file: RowFile::Training, row: row(3, 100.0, 130.0) },
            HeldOut { file: RowFile::Training, row: row(4, 130.0, 200.0) },
        ];
        let plan = ForecastPlan::new(partition(&rows), 70.0, &[
            PlanStream { obs: &o, training: &train, held_out: &held, design: DesignInput::NotSupplied },
        ]).unwrap();
        let got: Vec<(f64, f64, i64, f64)> = plan.targets().iter()
            .map(|t| (t.measures.start(), t.measures.end(), t.horizon, t.lead)).collect();
        assert_eq!(got, vec![(61.0, 92.0, 0, 31.0), (100.0, 130.0, 1, 69.0)]);
        assert_eq!(plan.sources()[0].1, TargetSource::HeldOutRows);
        assert!(TargetSource::HeldOutRows.describe().contains("lead"));
    }

    /// The target set is an input to the predictions, so it keys the
    /// forecast's address: splitting the held-out window [28, 35) of a
    /// window-column stream with no `emit_schedule` into [28, 31) and
    /// [31, 35) changes the targets, and so must change the address. The
    /// held-out outcomes do not.
    #[test]
    fn the_target_set_keys_the_forecasts_address() {
        let train = vec![row(0, 0.0, 14.0), row(1, 14.0, 28.0)];
        let rows = BTreeMap::from([("monthly".to_string(), train.clone())]);
        let o = monthly_windows();
        let address = |held: Vec<(f64, f64, f64)>| {
            let held: Vec<HeldOut> = held.into_iter().enumerate()
                .map(|(i, (a, b, y))| HeldOut { file: RowFile::Holdout, row: ObservedRow {
                    cell: Some(sim::inference::ObsCell::Scalar(y)), ..row(i, a, b) } })
                .collect();
            let plan = ForecastPlan::new(partition(&rows), 14.0, &[
                PlanStream { obs: &o, training: &train, held_out: &held, design: DesignInput::NotSupplied },
            ]).unwrap();
            crate::fit::forecast_output::forecast_options(&crate::fit::predict::PredictOptions::defaults(), &plan)
                .address_key("run")
        };
        let whole = address(vec![(28.0, 35.0, 7.0)]);
        assert_ne!(whole, address(vec![(28.0, 31.0, 7.0), (31.0, 35.0, 0.0)]), "a different target set");
        assert_ne!(whole, address(vec![(29.0, 35.0, 7.0)]), "one target, a different period");
        assert_eq!(whole, address(vec![(28.0, 35.0, 3.0)]), "the same targets, another outcome");
    }

    /// The held-out periods are validated: out of order, repeated,
    /// overlapping each other, or overlapping the training rows.
    #[test]
    fn held_out_periods_taken_as_targets_are_validated() {
        let train = vec![row(0, 0.0, 30.0), row(1, 30.0, 61.0)];
        let rows = BTreeMap::from([("monthly".to_string(), train.clone())]);
        let o = monthly_windows();
        let refuse = |held: Vec<(f64, f64)>, want: &str| {
            let held: Vec<HeldOut> = held.into_iter().enumerate()
                .map(|(i, (a, b))| HeldOut { file: RowFile::Holdout, row: row(i, a, b) }).collect();
            let err = ForecastPlan::new(partition(&rows), 200.0, &[
                PlanStream { obs: &o, training: &train, held_out: &held, design: DesignInput::NotSupplied },
            ]).unwrap_err();
            assert!(err.to_string().contains(want), "want {want:?}: {err}");
        };
        refuse(vec![(100.0, 130.0), (61.0, 92.0)], "out of order");
        refuse(vec![(61.0, 92.0), (61.0, 92.0)], "repeats");
        refuse(vec![(61.0, 92.0), (90.0, 120.0)], "overlaps held-out row 0");
        refuse(vec![(50.0, 92.0)], "overlaps its training row 1");
    }

    /// With no held-out rows the window-column stream still declares no
    /// rhythm (D42).
    #[test]
    fn a_window_column_stream_with_no_held_out_rows_is_still_refused() {
        let train = vec![row(0, 0.0, 30.0)];
        let rows = BTreeMap::from([("monthly".to_string(), train.clone())]);
        let o = monthly_windows();
        let err = ForecastPlan::new(partition(&rows), 60.0, &[
            PlanStream { obs: &o, training: &train, held_out: &[], design: DesignInput::NotSupplied },
        ]).unwrap_err();
        assert!(matches!(&err, PlanError::NoTargets { refusals } if matches!(refusals[..], [StreamRefusal::NoRhythm { .. }])), "{err:?}");
    }

    #[test]
    fn the_reach_parses_durations_in_the_models_unit() {
        assert_eq!(parse_reach("4 weeks", "days", None).unwrap(), 28.0);
        assert_eq!(parse_reach("28", "days", None).unwrap(), 28.0);
        assert_eq!(parse_reach("14 days", "weeks", None).unwrap(), 2.0);
        assert!(parse_reach("0 days", "days", None).is_err());
        assert!(parse_reach("four weeks", "days", None).is_err());
    }

    // Decision F (proposal 2026-11-16-calendar-durations-by-position.md):
    // the reach is counted from the forecast origin, the last training
    // observation, a date known only at run time; under a calendar origin a
    // month or year reach is refused with `--to`'s hint. Without an origin it
    // stays the average span.
    #[test]
    fn a_month_reach_is_refused_under_an_origin() {
        let err = parse_reach("6 months", "days", Some("2020-01-01")).unwrap_err();
        assert_eq!(err,
            "--horizon \"6 months\": calendar duration `6 months` cannot move the \
             forecast origin (the last training observation), whose date is known only \
             when the model runs. A calendar month from a date depends on which date it \
             is, and the one-step calendar rule resolves a step only from a date known \
             when the model compiles. State the offset in days: \"183 days\" \
             (≈ 6 months).");
        assert!(parse_reach("1 year", "days", Some("2020-01-01")).unwrap_err()
            .contains("\"365 days\" (≈ 1 year)"));
        assert_eq!(parse_reach("4 weeks", "days", Some("2020-01-01")).unwrap(), 28.0);
        assert_eq!(parse_reach("1 months", "days", None).unwrap(), 365.2425 / 12.0);
    }
}
