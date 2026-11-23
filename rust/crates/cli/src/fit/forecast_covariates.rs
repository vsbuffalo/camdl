//! What a forecast reads of a data-table forcing after its origin, and what
//! the user states those values are (`--covariate FORCING=STATEMENT`).
//!
//! A forcing read from a data table (births, population, a vaccination
//! schedule) is an input at every time, and after a forecast origin its values
//! are fixed in advance, observed later, or missing; camdl cannot tell which.
//! A forecast made with values observed after its origin is an after-the-fact
//! evaluation. So before any filter runs, [`classify`] decides for each forcing
//! of the model whether the forecast reads its table after the origin within
//! its reach, and [`Covariates::resolve`] refuses the forecast unless every
//! such forcing has a statement, and refuses a statement for a forcing that
//! needs none (`docs/camdl-run-spec.md`, "Forcings read after the origin").
//!
//! "After the origin" is judged on the table's argument: a forcing with a lag
//! `ℓ` reads its table at `t − ℓ`, so a run to the end of a forecast from
//! origin `o` with reach `h` reads arguments up to `o + h − ℓ`, and reads none
//! after `o` while `h ≤ ℓ`. A negative lag would read arguments after `o` from
//! model times before it; the compiler refuses one (E424), so only IR built
//! directly carries it. A lag that is a parameter is zero or more (E425 and the
//! parameter's kind range) but may take any such value over the draws, shorter
//! than the reach included, so every row after the origin is taken as read.
//! A cubic spline's value at every argument depends on every row, so a spline
//! table with a row after the origin is read after it whatever the lag.
//!
//! The statements enter the prediction's identity
//! ([`crate::fit::predict::ForecastKey::covariates`]). `hold` is applied to
//! the compiled models the forecast runs ([`Covariates::apply_holds`]): the
//! engine evaluates a held forcing as it is defined at and before the origin
//! and returns its value at the origin after it
//! ([`sim::compiled_model::HeldAfter`]), so the filtering before the origin is
//! the same, bit for bit, as without the hold.

use std::collections::BTreeMap;

use sim::compiled_model::CompiledTimeFuncKind;
use sim::resolved_expr::ResolvedExpr;

use crate::args::types::{CovariateSpec, CovariateStatement};
use crate::fit::runner::TIME_TIE_EPS;

/// What the label of a forecast reading covariates observed after its origin
/// says, on its manifest and (later) on every score of it.
pub(crate) const REALISED_LABEL: &str = "conditional on covariates observed after the origin";

/// How a table forcing passes between its rows (`interpolated`'s `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TableMethod {
    Linear,
    /// Each value held until the next row.
    Constant,
    /// A natural cubic spline through the rows: its value at every time
    /// depends on every row.
    Spline,
}

/// A forcing's lag, as the forecast judges it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Lag {
    None,
    /// A constant lag: a model time `t` reads the table at `t − ℓ`, after the
    /// origin for every `t > origin + ℓ`. A compiled model's constant lag is
    /// at least 0; a negative one comes only from IR built directly.
    Constant(f64),
    /// The lag is a parameter, at least 0 but different between draws and
    /// possibly shorter than the reach: every row of the table after the
    /// origin is taken as read.
    Parameter,
}

impl Lag {
    /// The last table argument a run to `reach_end` evaluates: `reach_end − ℓ`,
    /// and no bound for a parameter lag.
    fn last_argument(self, reach_end: f64) -> f64 {
        match self {
            Lag::None => reach_end,
            Lag::Constant(l) => reach_end - l,
            Lag::Parameter => f64::INFINITY,
        }
    }

    /// Why `hold` cannot be applied under this lag, if it cannot. A hold of
    /// the table from the origin changes only arguments after the origin. A
    /// model time `t` at or before the origin reads the argument `t − ℓ`,
    /// which is at or before the origin for every `ℓ ≥ 0`, so under no lag, a
    /// lag of 0 or more, or a parameter lag (at least 0), the hold leaves
    /// every value at and before the origin as it is. Only a negative lag
    /// reads the table after the origin from model times before it.
    fn refuses_hold(self) -> Option<String> {
        match self {
            // The compiler refuses a negative constant lag (E424); this arm
            // serves IR built directly.
            Lag::Constant(l) if l < 0.0 => Some(format!("its lag, {}, is negative", crate::quantile::fmt_time(l))),
            Lag::None | Lag::Constant(_) | Lag::Parameter => None,
        }
    }
}

/// A table forcing the forecast reads after its origin.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TableRead {
    pub method: TableMethod,
    /// The times of the table's first and last rows, in model time.
    pub first: f64,
    pub end: f64,
    pub lag: Lag,
    /// The `data = "path"` the forcing's table was read from, as written in
    /// the model; `None` for a table bound at run time or written inline.
    pub source: Option<String>,
    /// The SHA-256 of the table's rows as the forecast reads them: each row's
    /// time and value as the bits of an `f64`.
    pub rows_sha256: String,
    /// The forcing's value at the origin, the value `hold` carries forward.
    pub value_at_origin: f64,
}

/// Why a forcing needs no statement.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Unread {
    /// A periodic kind, defined for all time by the model.
    DefinedForAllTime { kind: &'static str },
    /// A `piecewise` forcing: its values are declared in the model, and it
    /// holds its last value after its last breakpoint. `in_reach` holds its
    /// breakpoints after the origin within the reach.
    Piecewise { in_reach: Vec<f64>, last: Option<f64> },
    /// A `bspline`: a curve with coefficients over a window, not a table.
    Window,
    /// A table whose last row is at or before the origin: the forecast reads
    /// no row of it after the origin.
    TableEndsByOrigin { end: f64 },
    /// A lagged table whose lag is at least the reach.
    WithinLag { lag: f64 },
    /// A step table (`method = constant`) whose next row after the origin is
    /// past the reach, so the forecast reads only the value at the origin.
    NoStepInReach { next: f64 },
}

/// One forcing of the model, as a forecast reads it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Reading {
    pub forcing: String,
    pub kind: ReadKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ReadKind {
    /// The forecast reads the table after the origin: a statement is required.
    Table(TableRead),
    NoStatement(Unread),
}

/// Each forcing of `compiled` (whose declarations are `model`'s), as a
/// forecast from `origin` to `reach_end` reads it. Every declared forcing is
/// classified, whether or not a rate reads it.
pub(crate) fn classify(model: &ir::Model, compiled: &sim::CompiledModel, origin: f64, reach_end: f64) -> Vec<Reading> {
    model
        .time_functions
        .iter()
        .filter_map(|decl| {
            let idx = *compiled.time_func_index.get(decl.name.as_str())?;
            let tf = &compiled.time_func_cache[idx];
            let lag = match &tf.lag {
                None => Lag::None,
                Some(ResolvedExpr::Const(l)) => Lag::Constant(*l),
                Some(_) => Lag::Parameter,
            };
            // The last table argument the run reads, up to the reach's end.
            let reach_arg = lag.last_argument(reach_end);
            // The breakpoints a note names are those within the reach in
            // model time, which for a parameter lag is judged with none.
            let note_arg = if lag == Lag::Parameter { reach_end } else { reach_arg };
            let after_origin = |t: f64| t > origin + TIME_TIE_EPS;
            let kind = match &tf.kind {
                CompiledTimeFuncKind::Sinusoidal { .. } => ReadKind::NoStatement(Unread::DefinedForAllTime { kind: "sinusoidal" }),
                CompiledTimeFuncKind::Periodic { .. } => ReadKind::NoStatement(Unread::DefinedForAllTime { kind: "periodic" }),
                CompiledTimeFuncKind::Fourier { .. } => ReadKind::NoStatement(Unread::DefinedForAllTime { kind: "fourier" }),
                CompiledTimeFuncKind::PeriodicSpline(_) => {
                    ReadKind::NoStatement(Unread::DefinedForAllTime { kind: "periodic_spline" })
                }
                CompiledTimeFuncKind::BSpline(_) => ReadKind::NoStatement(Unread::Window),
                CompiledTimeFuncKind::Piecewise { breakpoints, .. } => ReadKind::NoStatement(Unread::Piecewise {
                    in_reach: breakpoints.iter().copied().filter(|&b| after_origin(b) && b <= note_arg + TIME_TIE_EPS).collect(),
                    last: breakpoints.last().copied(),
                }),
                CompiledTimeFuncKind::Interpolated { times, values } => {
                    table(TableMethod::Linear, times, values, sim::propensity::interpolated_value(times, values, origin))
                }
                CompiledTimeFuncKind::Constant { times, values } => {
                    table(TableMethod::Constant, times, values, sim::propensity::constant_value(times, values, origin))
                }
                CompiledTimeFuncKind::CubicSpline(s) => table(TableMethod::Spline, &s.xs, &s.ys, s.eval(origin)),
            };
            // A table is read after the origin when it has a row after the
            // origin and the run's last argument passes the origin; a step
            // table only when one of those rows lies within the last argument.
            // A cubic spline's value at every argument depends on every row,
            // so one with a row after the origin is read after it whatever
            // the lag and the reach.
            let kind = match kind {
                ReadKind::Table(t) => {
                    let times = table_times(&tf.kind);
                    match times.iter().copied().find(|&x| after_origin(x)) {
                        None => ReadKind::NoStatement(Unread::TableEndsByOrigin { end: t.end }),
                        Some(_) if t.method != TableMethod::Spline && !after_origin(reach_arg) => {
                            let Lag::Constant(l) = lag else {
                                unreachable!("only a positive constant lag keeps the last argument at the origin")
                            };
                            ReadKind::NoStatement(Unread::WithinLag { lag: l })
                        }
                        Some(next) if t.method == TableMethod::Constant && next > reach_arg + TIME_TIE_EPS => {
                            ReadKind::NoStatement(Unread::NoStepInReach { next })
                        }
                        Some(_) => ReadKind::Table(TableRead { lag, source: decl.data_source.as_ref().map(|d| d.path.clone()), ..t }),
                    }
                }
                other => other,
            };
            Some(Reading { forcing: decl.name.clone(), kind })
        })
        .collect()
}

/// A table forcing whose values before a cut depend on its rows after it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreCutDependence {
    pub forcing: String,
    pub method: TableMethod,
    /// How many table rows after the cut the values before it depend on.
    pub rows_after_cut: usize,
}

/// Each table forcing of `compiled` (whose declarations are `model`'s) whose
/// values at model times from `t_start` up to `cut` depend on table rows after
/// `cut`. The cut is a fit's training extent or a forecast's origin; a fit or
/// forecast with such a forcing uses, before its cut, information from after
/// it. The judgement is on the table's argument, `t − lag`:
///
/// - a cubic spline depends on every row, so on every row after the cut,
///   whenever the run reads it at all;
/// - a linear table depends on the first row after the cut when the cut falls
///   strictly between two rows and the run reads an argument between the last
///   row before the cut and the cut;
/// - a step table (`method = constant`) depends only on rows at or before its
///   argument, so on rows after the cut only through a negative lag;
/// - a negative lag, which only IR built directly carries (the compiler
///   refuses one, E424), reads arguments after the cut from model times
///   before it, so the rows after the cut up to the last argument read count
///   (for a linear table, with the first row after that argument);
/// - a lag that is a parameter is at least 0, so the arguments read before
///   the cut are at or before it, and it is judged as a lag of 0, the value
///   whose arguments reach closest to the cut.
///
/// A label only: this changes no value and refuses nothing.
pub(crate) fn pre_cut_dependence(
    model: &ir::Model,
    compiled: &sim::CompiledModel,
    cut: f64,
    t_start: f64,
) -> Vec<PreCutDependence> {
    let after = |t: f64| t > cut + TIME_TIE_EPS;
    model
        .time_functions
        .iter()
        .filter_map(|decl| {
            let tf = &compiled.time_func_cache[*compiled.time_func_index.get(decl.name.as_str())?];
            let (method, times) = match &tf.kind {
                CompiledTimeFuncKind::Interpolated { times, .. } => (TableMethod::Linear, times.as_slice()),
                CompiledTimeFuncKind::Constant { times, .. } => (TableMethod::Constant, times.as_slice()),
                CompiledTimeFuncKind::CubicSpline(s) => (TableMethod::Spline, s.xs.as_slice()),
                _ => return None,
            };
            let rows_after = times.iter().filter(|&&x| after(x)).count();
            if rows_after == 0 || t_start > cut + TIME_TIE_EPS {
                return None;
            }
            let lag = match &tf.lag {
                None => Lag::None,
                Some(ResolvedExpr::Const(l)) => Lag::Constant(*l),
                Some(_) => Lag::Parameter,
            };
            // A parameter lag, at least 0, is judged as a lag of 0.
            let shift = if let Lag::Constant(l) = lag { l } else { 0.0 };
            // The arguments the model times [t_start, cut] read.
            let (lo, hi) = (t_start - shift, cut - shift);
            let n = match method {
                TableMethod::Spline => rows_after,
                TableMethod::Constant => times.iter().filter(|&&x| after(x) && x <= hi + TIME_TIE_EPS).count(),
                TableMethod::Linear if after(hi) => {
                    // A negative lag: the rows after the cut up to the
                    // last argument, and the row after that argument
                    // that the stretch before it interpolates toward.
                    let within = times.iter().filter(|&&x| after(x) && x <= hi + TIME_TIE_EPS).count();
                    let toward = times.iter().any(|&x| x > hi + TIME_TIE_EPS)
                        && !times.iter().any(|&x| (x - hi).abs() <= TIME_TIE_EPS);
                    within + usize::from(toward)
                }
                TableMethod::Linear => {
                    // The cut strictly between two rows, and an
                    // argument read between the row before it and it.
                    let before = times.iter().copied().rfind(|&x| x <= cut + TIME_TIE_EPS);
                    match before {
                        Some(prev) if !after(prev) && (prev - cut).abs() > TIME_TIE_EPS && hi > prev + TIME_TIE_EPS && lo <= hi => 1,
                        _ => 0,
                    }
                }
            };
            (n > 0).then(|| PreCutDependence { forcing: decl.name.clone(), method, rows_after_cut: n })
        })
        .collect()
}

impl PreCutDependence {
    /// The note a fit or forecast prints, naming the cut as `cut_name`.
    pub(crate) fn note(&self, cut_at: &str, cut_name: &str) -> String {
        format!(
            "forcing `{}` ({} table): values before the cut ({cut_at}, {cut_name}) depend on {} row{} after it",
            self.forcing,
            self.method.as_str(),
            self.rows_after_cut,
            if self.rows_after_cut == 1 { "" } else { "s" }
        )
    }

    /// The record a fit's `fit.meta.json` and a forecast's `forecast.json` hold.
    pub(crate) fn record(&self, cut: crate::run_meta::TimePoint) -> crate::run_meta::ForcingInterpolationRecord {
        crate::run_meta::ForcingInterpolationRecord {
            forcing: self.forcing.clone(),
            method: self.method.as_str().to_string(),
            cut,
            rows_after_cut: self.rows_after_cut,
        }
    }
}

impl TableMethod {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            TableMethod::Linear => "linear",
            TableMethod::Constant => "constant",
            TableMethod::Spline => "spline",
        }
    }
}

fn table(method: TableMethod, times: &[f64], values: &[f64], value_at_origin: f64) -> ReadKind {
    let mut rows = String::new();
    for (t, v) in times.iter().zip(values) {
        rows += &format!("{:016x}\t{:016x}\n", t.to_bits(), v.to_bits());
    }
    ReadKind::Table(TableRead {
        method,
        first: times.first().copied().unwrap_or(f64::NAN),
        end: times.last().copied().unwrap_or(f64::NAN),
        lag: Lag::None,
        source: None,
        rows_sha256: crate::hashing::sha256_hex(rows.as_bytes()),
        value_at_origin,
    })
}

fn table_times(kind: &CompiledTimeFuncKind) -> &[f64] {
    match kind {
        CompiledTimeFuncKind::Interpolated { times, .. } | CompiledTimeFuncKind::Constant { times, .. } => times,
        CompiledTimeFuncKind::CubicSpline(s) => &s.xs,
        _ => &[],
    }
}

/// How the messages of one forecast name its times.
pub(crate) struct ReachContext<'a> {
    pub origin: f64,
    /// `origin + horizon`.
    pub reach_end: f64,
    /// `--horizon` as written.
    pub as_written: &'a str,
    pub calendar: &'a ir::caltime::CalendarAxis,
}

impl ReachContext<'_> {
    /// A time as a message writes it: its date under a calendar origin,
    /// otherwise `t = …` in model time.
    fn at(&self, t: f64) -> String {
        match self.calendar.date_of(t) {
            Some(d) => d.into_string(),
            None => format!("t = {}", crate::quantile::fmt_time(t)),
        }
    }

    /// A duration in model time units.
    fn span(&self, d: f64) -> String {
        format!("{} {}", crate::quantile::fmt_time(d), self.calendar.time_unit.as_str())
    }
}

/// One stated forcing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Stated {
    pub forcing: String,
    pub statement: CovariateStatement,
    pub table: TableRead,
}

/// The statements of one forecast, and the notes on the forcings that need
/// none.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Covariates {
    /// In the model's order of forcings.
    pub stated: Vec<Stated>,
    pub notes: Vec<String>,
    /// The table forcings whose values before the origin depend on rows
    /// after it ([`pre_cut_dependence`]), whatever their statement.
    pub interpolation: Vec<crate::run_meta::ForcingInterpolationRecord>,
}

impl Covariates {
    /// The statements `specs` give for the forcings `readings` classify,
    /// checked: each names a forcing of the model, at most once, that the
    /// forecast reads after its origin, and `hold` names no cubic-spline
    /// table; and every such forcing has one. A forecast that breaks any of
    /// these is refused, with the message naming the forcing.
    pub(crate) fn resolve(readings: &[Reading], specs: &[CovariateSpec], ctx: &ReachContext) -> Result<Self, String> {
        let mut given: BTreeMap<&str, CovariateStatement> = BTreeMap::new();
        for s in specs {
            let Some(r) = readings.iter().find(|r| r.forcing == s.forcing) else {
                let names: Vec<&str> = readings.iter().map(|r| r.forcing.as_str()).collect();
                return Err(format!(
                    "--covariate {}={}: the model has no forcing `{}`{}.",
                    s.forcing,
                    s.statement.as_str(),
                    s.forcing,
                    if names.is_empty() {
                        "; it declares none".to_string()
                    } else {
                        format!("; its forcings are {}", names.join(", "))
                    }
                ));
            };
            if given.insert(s.forcing.as_str(), s.statement).is_some() {
                return Err(format!("--covariate names forcing `{}` twice; state it once.", s.forcing));
            }
            match &r.kind {
                ReadKind::NoStatement(why) => {
                    return Err(format!(
                        "--covariate {}={}: forcing `{}` needs no statement: {}. A statement is \
                         given only for a table the forecast reads after its origin; drop it.",
                        s.forcing,
                        s.statement.as_str(),
                        s.forcing,
                        describe_unread(why, ctx)
                    ))
                }
                ReadKind::Table(t) if s.statement == CovariateStatement::Hold => {
                    if let Some(why) = hold_refusal(t) {
                        return Err(format!(
                            "--covariate {f}=hold: forcing `{f}` cannot be held: {why}. State \
                             --covariate {f}=prospective or --covariate {f}=realised.",
                            f = s.forcing
                        ));
                    }
                }
                ReadKind::Table(_) => {}
            }
        }
        let missing: Vec<String> = readings
            .iter()
            .filter_map(|r| match &r.kind {
                ReadKind::Table(t) if !given.contains_key(r.forcing.as_str()) => Some(refusal(&r.forcing, t, ctx)),
                _ => None,
            })
            .collect();
        if !missing.is_empty() {
            return Err(missing.join("\nerror: "));
        }
        let stated = readings
            .iter()
            .filter_map(|r| match &r.kind {
                ReadKind::Table(t) => Some(Stated { forcing: r.forcing.clone(), statement: given[r.forcing.as_str()], table: t.clone() }),
                ReadKind::NoStatement(_) => None,
            })
            .collect();
        let notes = readings
            .iter()
            .filter_map(|r| match &r.kind {
                ReadKind::NoStatement(Unread::Piecewise { in_reach, last }) if !in_reach.is_empty() => {
                    Some(piecewise_note(&r.forcing, in_reach, *last, ctx))
                }
                _ => None,
            })
            .collect();
        Ok(Covariates { stated, notes, interpolation: Vec::new() })
    }

    /// The statements as the prediction's identity holds them, by forcing.
    /// Empty when the forecast reads no table forcing after its origin.
    pub(crate) fn key(&self) -> BTreeMap<String, CovariateStatement> {
        self.stated.iter().map(|s| (s.forcing.clone(), s.statement)).collect()
    }

    /// Whether any statement is `realised`: the forecast is conditional on
    /// covariates observed after its origin.
    pub(crate) fn conditional_on_realised(&self) -> bool {
        self.stated.iter().any(|s| s.statement == CovariateStatement::Realised)
    }

    /// Hold each forcing stated `hold` in `compiled` at its value at the
    /// origin for every argument after the origin.
    pub(crate) fn apply_holds(&self, compiled: &mut sim::CompiledModel, origin: f64) -> Result<(), String> {
        for s in self.stated.iter().filter(|s| s.statement == CovariateStatement::Hold) {
            let value = compiled
                .hold_forcing_after(&s.forcing, origin)
                .map_err(|e| format!("holding forcing `{}` at the origin: {e}", s.forcing))?;
            if value.to_bits() != s.table.value_at_origin.to_bits() {
                return Err(format!(
                    "internal error: forcing `{}` is held at {value}, but the plan read its value at \
                     the origin as {}",
                    s.forcing, s.table.value_at_origin
                ));
            }
        }
        Ok(())
    }

    /// The `covariates` block of `forecast.json`; `None` when there is
    /// nothing to say, so a forecast that reads no table forcing after its
    /// origin writes no block.
    pub(crate) fn manifest(&self, origin: f64, point: impl Fn(f64) -> serde_json::Value) -> Option<serde_json::Value> {
        if self.stated.is_empty() && self.notes.is_empty() && self.interpolation.is_empty() {
            return None;
        }
        let forcings: Vec<serde_json::Value> = self
            .stated
            .iter()
            .map(|s| {
                let mut f = serde_json::json!({
                    "forcing": s.forcing,
                    "statement": s.statement,
                    "method": s.table.method,
                    "table": {
                        "source": s.table.source,
                        "rows_sha256": s.table.rows_sha256,
                        "first": point(s.table.first),
                        "end": point(s.table.end),
                    },
                    "lag": match s.table.lag {
                        Lag::None => serde_json::Value::Null,
                        Lag::Constant(l) => serde_json::json!(l),
                        Lag::Parameter => serde_json::json!("parameter"),
                    },
                });
                if s.statement == CovariateStatement::Hold {
                    // The hold is on the table's argument: the forcing is held
                    // for every model time t with t − lag after the origin.
                    f["held"] = serde_json::json!({
                        "after": point(origin),
                        "on": "the table's argument, t − lag",
                        "value": s.table.value_at_origin,
                    });
                }
                f
            })
            .collect();
        let mut doc = serde_json::json!({ "forcings": forcings, "notes": self.notes });
        if !self.interpolation.is_empty() {
            doc["interpolation"] = serde_json::to_value(&self.interpolation).expect("a record serializes");
        }
        if self.conditional_on_realised() {
            doc["label"] = serde_json::json!(REALISED_LABEL);
        }
        Some(doc)
    }

    /// One line per statement, for the command's stderr.
    pub(crate) fn describe(&self) -> Vec<String> {
        self.stated
            .iter()
            .map(|s| {
                let what = match s.statement {
                    CovariateStatement::Prospective => {
                        "read as it is: its values after the origin were fixed or assumed at the origin".to_string()
                    }
                    CovariateStatement::Realised => format!("read as it is: the forecast is {REALISED_LABEL}"),
                    CovariateStatement::Hold => {
                        format!(
                            "held at {}, the table's value at the origin, wherever its argument t − lag is after the origin",
                            crate::quantile::fmt_value(s.table.value_at_origin)
                        )
                    }
                };
                format!("forcing `{}` after the forecast origin: {what}", s.forcing)
            })
            .chain(self.notes.iter().cloned())
            .collect()
    }
}

/// The refusal of a forecast that reads the table of `forcing` after its
/// origin with no statement.
fn refusal(forcing: &str, t: &TableRead, ctx: &ReachContext) -> String {
    let source = t.source.as_ref().map(|s| format!(" `{s}`")).unwrap_or_default();
    let mut msg = format!(
        "forcing `{forcing}` is read after the forecast origin and nothing says how.\n  \
         origin {}, reach {} (to {}); its table{source} has rows to {}.\n",
        ctx.at(ctx.origin),
        ctx.as_written,
        ctx.at(ctx.reach_end),
        ctx.at(t.end),
    );
    match t.lag {
        Lag::None => {}
        Lag::Constant(l) => msg += &format!(
            "  It has a lag of {}, so the forecast reads its table at t − lag, after the origin up to {}.\n",
            ctx.span(l),
            ctx.at(ctx.reach_end - l)
        ),
        Lag::Parameter => msg += "  Its lag is a parameter, which may be shorter than the reach, so every \
                                  row of its table after the origin is taken as read.\n",
    }
    let hold = hold_refusal(t);
    let mut options = vec![
        ("prospective", "the values after the origin were fixed or assumed at the origin"),
        ("realised", "they were observed after the origin"),
    ];
    if hold.is_none() {
        options.push(("hold", "carry the value at the origin forward"));
    }
    let width = options.iter().map(|(s, _)| s.len()).max().unwrap_or(0);
    msg += "  State one of:";
    for (s, what) in &options {
        msg += &format!("\n    --covariate {forcing}={s:<width$}   {what}");
    }
    if let Some(why) = hold {
        msg += &format!("\n  (`hold` is not offered: {why}.)");
    }
    msg
}

/// Why a table forcing cannot be held at the origin, if it cannot: a hold
/// must leave every value at and before the origin, in model time, as it is.
fn hold_refusal(t: &TableRead) -> Option<String> {
    if t.method == TableMethod::Spline {
        return Some(
            "it is a cubic spline through its table (method = spline), whose value at every time depends \
             on every row, so cutting the table at the origin would change the forcing before the origin too"
                .to_string(),
        );
    }
    t.lag.refuses_hold().map(|lag| {
        format!(
            "{lag}, so model times at or before the origin read its table after the origin, and holding \
             the table from the origin would change the forcing before the origin"
        )
    })
}

fn describe_unread(why: &Unread, ctx: &ReachContext) -> String {
    match why {
        Unread::DefinedForAllTime { kind } => format!("it is a {kind} forcing, defined by the model for all time"),
        Unread::Piecewise { .. } => "it is a piecewise forcing, whose values the model declares, and \
                                     which holds its last value after its last breakpoint"
            .to_string(),
        Unread::Window => "it is a bspline, a curve with coefficients rather than a table; past its \
                           window it is refused as forcing-outside-range"
            .to_string(),
        Unread::TableEndsByOrigin { end } => format!(
            "its table ends at {}, at or before the origin, so the forecast reads no row of it after \
             the origin (past its end it is refused as forcing-outside-range, or held under \
             --allow-forcing-extrapolation)",
            ctx.at(*end)
        ),
        Unread::WithinLag { lag } => format!(
            "its lag, {}, is at least the reach, so the forecast reads its table only up to the origin",
            ctx.span(*lag)
        ),
        Unread::NoStepInReach { next } => format!(
            "it is read as a step (method = constant), and its first row after the origin, at {}, is \
             past the reach, so the forecast reads only its value at the origin",
            ctx.at(*next)
        ),
    }
}

/// The note on a piecewise forcing whose breakpoints the forecast passes.
fn piecewise_note(forcing: &str, in_reach: &[f64], last: Option<f64>, ctx: &ReachContext) -> String {
    let at: Vec<String> = in_reach.iter().map(|&b| ctx.at(b)).collect();
    let holds = match (last, in_reach.last()) {
        (Some(l), Some(&b)) if l == b => ", and holds its last value from the last of them on",
        _ => "",
    };
    format!(
        "forcing `{forcing}` is piecewise and changes at {} after the forecast origin, within the \
         reach, as the model declares{holds}",
        at.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::expr::Expr;
    use ir::time_func::{InterpMethod, Interpolated, Periodic, Piecewise, Sinusoidal, TimeFuncKind, TimeFunction};

    fn forcing(name: &str, kind: TimeFuncKind, lag: Option<f64>) -> TimeFunction {
        TimeFunction { name: name.into(), kind, dim: (0, 0), lag: lag.map(Expr::const_), data_source: None }
    }

    /// A table through the rows (0, 1), (20, 2), (40, 3), (60, 4), (100, 5).
    fn table(method: InterpMethod) -> TimeFuncKind {
        let t = [0.0, 20.0, 40.0, 60.0, 100.0].map(Expr::const_).to_vec();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0].map(Expr::const_).to_vec();
        TimeFuncKind::Interpolated(Interpolated::knots(t, v, method))
    }

    /// The SIR golden with `forcings` declared, compiled.
    fn model(forcings: Vec<TimeFunction>) -> (ir::Model, sim::CompiledModel) {
        let path = format!("{}/../../../ocaml/golden/sir_basic.ir.json", env!("CARGO_MANIFEST_DIR"));
        let (mut m, _) = crate::util::load_ir(&path).unwrap();
        for p in &mut m.parameters {
            p.value = p.value.with_value(0.5);
        }
        m.time_functions.extend(forcings);
        let compiled = sim::CompiledModel::new(m.clone()).unwrap();
        (m, compiled)
    }

    fn readings(forcings: Vec<TimeFunction>, origin: f64, reach_end: f64) -> Vec<Reading> {
        let (m, c) = model(forcings);
        classify(&m, &c, origin, reach_end)
    }

    fn kind_of<'a>(r: &'a [Reading], name: &str) -> &'a ReadKind {
        &r.iter().find(|r| r.forcing == name).unwrap().kind
    }

    fn ctx<'a>(calendar: &'a ir::caltime::CalendarAxis, as_written: &'a str, origin: f64, h: f64) -> ReachContext<'a> {
        ReachContext { origin, reach_end: origin + h, as_written, calendar }
    }

    fn spec(s: &str) -> CovariateSpec {
        s.parse().unwrap()
    }

    /// A forecast from day 24 reaching 2 weeks reads a table with rows to day
    /// 100 after its origin, and is refused with no statement. The message
    /// names the forcing, the origin, the reach and the table's last row, as
    /// calendar dates when the model has an origin and as model time
    /// otherwise, and the three statements.
    #[test]
    fn a_table_read_after_the_origin_is_refused_with_its_dates() {
        let r = readings(vec![forcing("pop", table(InterpMethod::Linear), None)], 24.0, 38.0);
        assert!(matches!(kind_of(&r, "pop"), ReadKind::Table(_)));
        let dated = ir::caltime::CalendarAxis::parse("days", Some("2020-01-01")).unwrap();
        let err = Covariates::resolve(&r, &[], &ctx(&dated, "2 weeks", 24.0, 14.0)).unwrap_err();
        for want in [
            "forcing `pop` is read after the forecast origin and nothing says how.",
            "origin 2020-01-25, reach 2 weeks (to 2020-02-08); its table has rows to 2020-04-10.",
            "--covariate pop=prospective   the values after the origin were fixed or assumed at the origin",
            "--covariate pop=realised      they were observed after the origin",
            "--covariate pop=hold          carry the value at the origin forward",
        ] {
            assert!(err.contains(want), "missing {want:?} in:\n{err}");
        }
        let undated = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let err = Covariates::resolve(&r, &[], &ctx(&undated, "14", 24.0, 14.0)).unwrap_err();
        assert!(err.contains("origin t = 24, reach 14 (to t = 38); its table has rows to t = 100."), "{err}");
    }

    /// Each statement resolves; the key holds it, and `realised` alone makes
    /// the forecast conditional on covariates observed after its origin, which
    /// its manifest block says.
    #[test]
    fn each_statement_resolves_and_realised_labels_the_forecast() {
        let r = readings(vec![forcing("pop", table(InterpMethod::Linear), None)], 24.0, 38.0);
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let point = |t: f64| serde_json::json!({ "t": t });
        for (s, statement) in [
            ("pop=prospective", CovariateStatement::Prospective),
            ("pop=realised", CovariateStatement::Realised),
            ("pop=hold", CovariateStatement::Hold),
        ] {
            let c = Covariates::resolve(&r, &[spec(s)], &ctx(&cal, "14", 24.0, 14.0)).unwrap();
            assert_eq!(c.key(), BTreeMap::from([("pop".to_string(), statement)]));
            let block = c.manifest(24.0, point).unwrap();
            assert_eq!(block["forcings"][0]["statement"], statement.as_str());
            let realised = statement == CovariateStatement::Realised;
            assert_eq!(c.conditional_on_realised(), realised);
            assert_eq!(block.get("label").and_then(|l| l.as_str()), realised.then_some(REALISED_LABEL), "{s}");
            assert_eq!(block["forcings"][0].get("held").is_some(), statement == CovariateStatement::Hold);
        }
    }

    /// A forcing with a lag of 10 days reads its table at t − 10: a reach of
    /// 7 days reads it only up to the origin and needs no statement, and a
    /// reach of 14 days reads it after the origin and needs one.
    #[test]
    fn a_lagged_table_needs_a_statement_only_once_the_reach_exceeds_its_lag() {
        let lagged = || vec![forcing("pop", table(InterpMethod::Linear), Some(10.0))];
        let short = readings(lagged(), 24.0, 31.0);
        assert_eq!(kind_of(&short, "pop"), &ReadKind::NoStatement(Unread::WithinLag { lag: 10.0 }));
        let long = readings(lagged(), 24.0, 38.0);
        let ReadKind::Table(t) = kind_of(&long, "pop") else { panic!("a reach past the lag reads the table") };
        assert_eq!(t.lag, Lag::Constant(10.0));
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let err = Covariates::resolve(&long, &[], &ctx(&cal, "14", 24.0, 14.0)).unwrap_err();
        assert!(
            err.contains("It has a lag of 10 days, so the forecast reads its table at t − lag, after the origin up to t = 28."),
            "{err}"
        );
        let refused = Covariates::resolve(&short, &[spec("pop=hold")], &ctx(&cal, "7", 24.0, 7.0)).unwrap_err();
        assert!(refused.contains("needs no statement") && refused.contains("lag"), "{refused}");
    }

    /// Periodic kinds and a piecewise forcing need no statement, and a
    /// statement for one is refused; a piecewise breakpoint within the reach
    /// is named in a note.
    #[test]
    fn periodic_and_piecewise_forcings_need_no_statement() {
        let c = Expr::const_;
        let r = readings(
            vec![
                forcing(
                    "season",
                    TimeFuncKind::Sinusoidal(Sinusoidal { amplitude: c(0.2), period: c(365.0), phase: c(0.0), baseline: c(1.0) }),
                    None,
                ),
                forcing("week", TimeFuncKind::Periodic(Periodic { period: c(7.0), values: vec![c(1.0), c(0.5)] }), None),
                forcing(
                    "lockdown",
                    TimeFuncKind::Piecewise(Piecewise { breakpoints: vec![c(10.0), c(30.0)], values: vec![c(1.0), c(0.4), c(0.8)] }),
                    None,
                ),
            ],
            24.0,
            38.0,
        );
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let resolved = Covariates::resolve(&r, &[], &ctx(&cal, "14", 24.0, 14.0)).unwrap();
        assert!(resolved.key().is_empty(), "nothing needs a statement");
        assert_eq!(resolved.notes.len(), 1, "{:?}", resolved.notes);
        assert!(
            resolved.notes[0].contains("`lockdown`") && resolved.notes[0].contains("t = 30") && !resolved.notes[0].contains("t = 10"),
            "the note names the breakpoint the reach passes: {}",
            resolved.notes[0]
        );
        for name in ["season", "week", "lockdown"] {
            let err =
                Covariates::resolve(&r, &[spec(&format!("{name}=realised"))], &ctx(&cal, "14", 24.0, 14.0)).unwrap_err();
            assert!(err.contains(&format!("forcing `{name}` needs no statement")), "{err}");
        }
    }

    /// A table whose last row is at the origin, and a step table whose next
    /// row is past the reach, read no table value after the origin.
    #[test]
    fn a_table_read_only_up_to_the_origin_needs_no_statement() {
        let ends = |m: InterpMethod| {
            let t = [0.0, 12.0, 24.0].map(Expr::const_).to_vec();
            let v = [1.0, 2.0, 3.0].map(Expr::const_).to_vec();
            TimeFuncKind::Interpolated(Interpolated::knots(t, v, m))
        };
        let r = readings(
            vec![forcing("ended", ends(InterpMethod::Linear), None), forcing("step", table(InterpMethod::Constant), None)],
            24.0,
            38.0,
        );
        assert_eq!(kind_of(&r, "ended"), &ReadKind::NoStatement(Unread::TableEndsByOrigin { end: 24.0 }));
        assert_eq!(kind_of(&r, "step"), &ReadKind::NoStatement(Unread::NoStepInReach { next: 40.0 }));
        let r = readings(vec![forcing("step", table(InterpMethod::Constant), None)], 24.0, 44.0);
        assert!(matches!(kind_of(&r, "step"), ReadKind::Table(_)), "a reach past the next step reads it");
    }

    /// `hold` is refused for a cubic-spline table, which the refusal of a
    /// missing statement does not offer it for; an unknown forcing and a
    /// forcing stated twice are refused.
    #[test]
    fn hold_is_refused_for_a_spline_table_and_a_statement_names_one_forcing_once() {
        let r = readings(vec![forcing("pop", table(InterpMethod::Spline), None)], 24.0, 38.0);
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let c = ctx(&cal, "14", 24.0, 14.0);
        let err = Covariates::resolve(&r, &[spec("pop=hold")], &c).unwrap_err();
        assert!(err.contains("--covariate pop=hold") && err.contains("cubic spline"), "{err}");
        let missing = Covariates::resolve(&r, &[], &c).unwrap_err();
        assert!(!missing.contains("--covariate pop=hold ") && missing.contains("`hold` is not offered"), "{missing}");
        assert!(Covariates::resolve(&r, &[spec("pop=realised")], &c).is_ok());
        let err = Covariates::resolve(&r, &[spec("births=realised")], &c).unwrap_err();
        assert!(err.contains("no forcing `births`") && err.contains("pop"), "{err}");
        let err = Covariates::resolve(&r, &[spec("pop=realised"), spec("pop=prospective")], &c).unwrap_err();
        assert!(err.contains("twice"), "{err}");
    }

    /// A spline table with a lag of 20 days and a reach of 14: the run's last
    /// argument stays before the origin, but a natural spline's value at every
    /// argument depends on every row, so its rows after the origin are read
    /// and it needs a statement; `hold` is not offered for it.
    #[test]
    fn a_spline_table_within_its_lag_still_needs_a_statement() {
        let r = readings(vec![forcing("pop", table(InterpMethod::Spline), Some(20.0))], 24.0, 38.0);
        let ReadKind::Table(t) = kind_of(&r, "pop") else { panic!("a spline with rows after the origin is read after it") };
        assert_eq!(t.lag, Lag::Constant(20.0));
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let c = ctx(&cal, "14", 24.0, 14.0);
        let err = Covariates::resolve(&r, &[], &c).unwrap_err();
        assert!(err.contains("forcing `pop` is read after the forecast origin") && err.contains("`hold` is not offered"), "{err}");
        assert!(Covariates::resolve(&r, &[spec("pop=hold")], &c).unwrap_err().contains("cubic spline"));
        assert!(Covariates::resolve(&r, &[spec("pop=prospective")], &c).is_ok());
    }

    /// A step table with rows at 0, 20, 40 and 63, from an origin at day 24
    /// with a reach of 14 days. With no lag its next row, day 40, is past the
    /// reach. A lag of −5 days (IR built directly; the compiler refuses one)
    /// reads arguments up to day 43, and a lag that is a parameter may be
    /// shorter than the reach, so both are taken to read the day-40 row and
    /// need a statement. The negative lag cannot be held, since model times
    /// before the origin read the table after it; the parameter lag, at least
    /// 0, can.
    #[test]
    fn a_negative_or_parameter_lag_reads_rows_after_the_origin_and_only_the_negative_cannot_be_held() {
        let step = || {
            let t = [0.0, 20.0, 40.0, 63.0].map(Expr::const_).to_vec();
            let v = [1.0, 2.0, 3.0, 4.0].map(Expr::const_).to_vec();
            TimeFuncKind::Interpolated(Interpolated::knots(t, v, InterpMethod::Constant))
        };
        let none = readings(vec![forcing("pop", step(), None)], 24.0, 38.0);
        assert_eq!(kind_of(&none, "pop"), &ReadKind::NoStatement(Unread::NoStepInReach { next: 40.0 }));
        let negative = readings(vec![forcing("pop", step(), Some(-5.0))], 24.0, 38.0);
        let parameter = readings(
            vec![TimeFunction { lag: Some(Expr::param("gamma")), ..forcing("pop", step(), None) }],
            24.0,
            38.0,
        );
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let c = ctx(&cal, "14", 24.0, 14.0);
        for (what, r, lag) in [("negative", &negative, Lag::Constant(-5.0)), ("parameter", &parameter, Lag::Parameter)] {
            let ReadKind::Table(t) = kind_of(r, "pop") else { panic!("{what}: the day-40 row is read") };
            assert_eq!(t.lag, lag);
            assert!(Covariates::resolve(r, &[spec("pop=realised")], &c).is_ok(), "{what}");
        }
        let err = Covariates::resolve(&negative, &[], &c).unwrap_err();
        assert!(err.contains("`hold` is not offered") && !err.contains("pop=hold "), "{err}");
        let err = Covariates::resolve(&negative, &[spec("pop=hold")], &c).unwrap_err();
        assert!(err.contains("cannot be held") && err.contains("negative") && err.contains("before the origin"), "{err}");
        let err = Covariates::resolve(&parameter, &[], &c).unwrap_err();
        assert!(err.contains("pop=hold ") && !err.contains("`hold` is not offered") && !err.contains("negative"), "{err}");
        assert!(Covariates::resolve(&parameter, &[spec("pop=hold")], &c).is_ok());
    }

    /// The refusal's statements line up in one column whatever the forcing's
    /// name.
    #[test]
    fn the_refusals_statements_line_up() {
        let r = readings(vec![forcing("a_long_forcing_name_p0", table(InterpMethod::Linear), None)], 24.0, 38.0);
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let err = Covariates::resolve(&r, &[], &ctx(&cal, "14", 24.0, 14.0)).unwrap_err();
        let columns: Vec<usize> = err
            .lines()
            .filter(|l| l.trim_start().starts_with("--covariate"))
            .map(|l| {
                // Where the explanation starts: the first word after the statement.
                let eq = l.find('=').unwrap();
                let gap = eq + l[eq..].find(' ').unwrap();
                gap + l[gap..].find(|c: char| c != ' ').unwrap()
            })
            .collect();
        assert_eq!(columns.len(), 3, "{err}");
        assert!(columns.iter().all(|&c| c == columns[0]), "{err}");
    }

    /// The rows after a cut at `cut` that each forcing's values before it
    /// depend on, for a run from day 0.
    fn dependence(forcings: Vec<TimeFunction>, cut: f64) -> Vec<(String, usize)> {
        let (m, c) = model(forcings);
        pre_cut_dependence(&m, &c, cut, 0.0).into_iter().map(|d| (d.forcing, d.rows_after_cut)).collect()
    }

    /// A cut at day 50 of the table with rows at 0, 20, 40, 60 and 100: a
    /// spline depends on both rows after it; a linear table, whose cut falls
    /// between the rows at 40 and 60, on the row at 60; a step table on
    /// none. A linear table cut at one of its rows, and one ending at the
    /// cut, depend on none.
    #[test]
    fn values_before_the_cut_depend_on_rows_after_it_for_splines_and_linear_tables_cut_between_rows() {
        assert_eq!(dependence(vec![forcing("pop", table(InterpMethod::Spline), None)], 50.0), vec![("pop".to_string(), 2)]);
        assert_eq!(dependence(vec![forcing("pop", table(InterpMethod::Linear), None)], 50.0), vec![("pop".to_string(), 1)]);
        assert_eq!(dependence(vec![forcing("pop", table(InterpMethod::Constant), None)], 50.0), vec![]);
        assert_eq!(dependence(vec![forcing("pop", table(InterpMethod::Linear), None)], 40.0), vec![], "a cut at a row");
        let ends = TimeFuncKind::Interpolated(Interpolated::knots(
            [0.0, 12.0, 24.0].map(Expr::const_).to_vec(),
            [1.0, 2.0, 3.0].map(Expr::const_).to_vec(),
            InterpMethod::Spline,
        ));
        assert_eq!(dependence(vec![forcing("pop", ends, None)], 24.0), vec![], "a table ending at the cut");
    }

    /// A lag shifts the arguments the run before the cut reads. A positive
    /// lag can keep them before the stretch the next row reaches into; a
    /// negative one (IR built directly) reads rows after the cut, even of a
    /// step table; a lag that is a parameter, at least 0, is judged as a lag
    /// of 0.
    #[test]
    fn a_lag_judges_the_dependence_on_the_tables_argument() {
        let linear = |lag| vec![forcing("pop", table(InterpMethod::Linear), lag)];
        assert_eq!(dependence(linear(Some(15.0)), 50.0), vec![], "arguments up to 35 read no row after 40");
        assert_eq!(dependence(linear(Some(5.0)), 50.0), vec![("pop".to_string(), 1)], "arguments up to 45 interpolate toward 60");
        assert_eq!(dependence(linear(Some(-5.0)), 50.0), vec![("pop".to_string(), 1)]);
        assert_eq!(dependence(linear(Some(-15.0)), 50.0), vec![("pop".to_string(), 2)], "the row at 60, and 100 beyond the argument 65");
        let step = |lag| vec![forcing("pop", table(InterpMethod::Constant), lag)];
        assert_eq!(dependence(step(Some(-15.0)), 50.0), vec![("pop".to_string(), 1)]);
        let parameter = |m| vec![TimeFunction { lag: Some(Expr::param("gamma")), ..forcing("pop", table(m), None) }];
        assert_eq!(dependence(parameter(InterpMethod::Constant), 50.0), vec![], "a step table reads no row after the cut");
        assert_eq!(dependence(parameter(InterpMethod::Linear), 50.0), vec![("pop".to_string(), 1)], "as with no lag");
        assert_eq!(dependence(parameter(InterpMethod::Spline), 50.0), vec![("pop".to_string(), 2)]);
    }

    /// Holding applies to the compiled model each forcing stated `hold`, at
    /// the value the plan read at the origin, and to no other.
    #[test]
    fn apply_holds_holds_only_the_forcings_stated_hold() {
        let forcings =
            || vec![forcing("pop", table(InterpMethod::Linear), None), forcing("births", table(InterpMethod::Constant), None)];
        let r = readings(forcings(), 30.0, 50.0);
        let cal = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        let c = Covariates::resolve(&r, &[spec("pop=hold"), spec("births=prospective")], &ctx(&cal, "20", 30.0, 20.0)).unwrap();
        let (_, mut compiled) = model(forcings());
        c.apply_holds(&mut compiled, 30.0).unwrap();
        let held = |n: &str| compiled.time_func_cache[compiled.time_func_index[n]].held;
        assert_eq!(held("pop"), Some(sim::compiled_model::HeldAfter { from: 30.0, value: 2.5 }));
        assert_eq!(held("births"), None);
    }
}
