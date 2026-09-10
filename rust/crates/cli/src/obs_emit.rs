//! What a simulated observation file carries for one stream: which rows, under
//! which temporal columns, each drawn over which period (gh#833, ruling 3 of
//! the observation-time proposal), and — for a dataset simulated on a fit's own
//! observation design — which rows are holes (gh#831).
//!
//! The emitter follows the stream's own declaration, so the file it writes is
//! the file the loader reads back. There are two ways a stream's rows are
//! fixed, and they meet in one [`EmitPlan`], one writer and one sampler:
//!
//! - **From the model's declaration** ([`plan_emission`]), for a run with no
//!   data bound. The label is the emit-schedule time; the value is the flow
//!   over whatever period the stream's `covers` assigns to that label — the
//!   same arithmetic the loader applies to a label it reads
//!   (`Covers::period_of`) — and a row whose period does not lie within the run
//!   is not written, because the run never simulated it. A stream declared with
//!   `window_start`/`window_stop` gets contiguous windows closing at the emit
//!   times, written under the declared column names.
//!
//!   One consequence is visible: under `closing_at` with a schedule starting at
//!   `t_start`, the first emit time's period `[t_start − Δ, t_start)` falls
//!   outside the run and is dropped — a row nothing could score, so no
//!   likelihood moves. An accumulating stream whose IR carries no declaration
//!   has no reading to write under and is refused (the compiler never produces
//!   one, E350).
//!
//! - **From a fit's bound data** ([`plan_bound_rows`]), the design-preserving
//!   case gh#831 asks for. The rows are the ones the loader gave the fit: each
//!   observed label, each row's own period — including per-row
//!   `window_start`/`window_stop` widths the model has no rule to generate —
//!   and each `NA` hole, which stays a row carrying its period and no value. A
//!   dataset drawn on this design carries exactly the information the real one
//!   does, which is the property the simulation-based self-consistency test
//!   depends on: simulating on a regular grid would give the synthetic fit more
//!   information than the real fit has.
//!
//! A stream whose likelihood reads a data column is refused by name (gh#829),
//! with one exception: a `binomial`/`beta_binomial` `n` over a ratio-of-flows
//! projection (proposal 2026-09-09). That denominator is a quantity the model
//! generates — the ratio's own denominator flow over the row, the events that
//! were classified — so the emitter writes the column from the model, rounded
//! to the count it is, and draws `k ~ binomial(n, ratio)`; the file re-loads
//! with the design's information and no more. A survey's `tested` is
//! surveillance effort the model has no term for and stays refused.
//!
//! [`simulate_dataset`] runs the forward model once and writes the dataset
//! either way.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ir::observation::{ColumnRole, Covers, ObservationModel, TemporalKind};
use sim::compiled_model::CompiledModel;
use sim::inference::{Coverage, ObsCell, StreamTimes};
use sim::rng::StatefulRng;

/// The temporal column(s) an emitted file carries for a stream — the ones the
/// stream declared in `columns { }`, so the file re-loads under its own model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TemporalColumns {
    /// One label column, under the stream's `: time` column name.
    Time(String),
    /// A `window_start` / `window_stop` pair, under their declared names.
    Window { start: String, stop: String },
}

/// How an emitted file spells its temporal cells.
///
/// A file bound as a design was published in one of two representations, and
/// the design-preserving writer gives it back in the one it came in as
/// (gh#882): a stream whose cells were ISO dates is re-emitted as ISO dates
/// through the model's `origin` and `time_unit`, and one whose cells were
/// numbers keeps the numbers. Rows planned from the model's own declaration
/// have no input file to have been written in, and stay numeric.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TemporalFormat {
    /// Internal model time, written as the number it is.
    Numeric,
    /// ISO dates, rendered through the model's calendar anchor.
    Dated { origin: String, time_unit: String },
}

impl TemporalFormat {
    /// One temporal cell, as this format spells it.
    ///
    /// A dated cell must name the same instant it renders from, or the file
    /// would re-load as a different design — so the rendered date is converted
    /// back and compared before it is written. A boundary that is not a whole
    /// day (a `covers` span of half a day, say) has no ISO-date spelling, and
    /// is an error here rather than a cell silently snapped to midnight.
    fn cell(&self, t: f64) -> Result<String, String> {
        match self {
            TemporalFormat::Numeric => Ok(format!("{t}")),
            TemporalFormat::Dated { origin, time_unit } => {
                let date = ir::caltime::internal_to_date(origin, t, time_unit)
                    .map_err(|e| format!(
                        "cannot write t = {t} as a date under origin {origin}: {e:?}"))?;
                let back = ir::caltime::date_to_internal(origin, &date, time_unit)
                    .map_err(|e| format!("{date}: {e:?}"))?;
                if (back - t).abs() > 1e-9 {
                    return Err(format!(
                        "the temporal cell t = {t} falls between calendar days: the nearest \
                         date, {date}, is t = {back} under origin {origin} and time_unit \
                         '{time_unit}'. A dated file cannot state it, so writing one would \
                         re-load as a different design."));
                }
                Ok(date)
            }
        }
    }
}

/// One row the emitter writes: what its temporal column(s) say, what the
/// value is drawn over, and whether the row carries a value at all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EmitRow {
    /// The `: time` column's value — the emit-schedule time, or the bound
    /// row's own label. For a windowed stream this is the window's stop; the
    /// start is in `coverage`.
    pub label: f64,
    /// An instant at `label`, or the flow over `[start, stop)`.
    pub coverage: Coverage,
    /// `false` for a row the design has as a hole: its period is written and
    /// its value is the loader's `NA` token. Dropping the row instead would
    /// hand the next row a wider window than the fit sees, so a hole is a row.
    /// Every row planned from a declaration is observed — only a bound design
    /// has holes.
    pub observed: bool,
}

/// Everything a writer needs to emit one stream's file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EmitPlan {
    pub columns: TemporalColumns,
    /// How the temporal column(s) are spelled — the representation the bound
    /// file used (gh#882).
    pub format: TemporalFormat,
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
            .map(|&t| EmitRow { label: t, coverage: Coverage::Instant, observed: true })
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
                    rows.push(EmitRow {
                        label: t,
                        coverage: Coverage::Interval { start, stop },
                        observed: true,
                    });
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
                    rows.push(EmitRow {
                        label: t,
                        coverage: Coverage::Interval { start: prev, stop: t },
                        observed: true,
                    });
                }
                prev = t;
            }
            rows
        }
    };
    Ok(EmitPlan { columns, format: TemporalFormat::Numeric, scored, rows })
}

/// Plan the rows a stream's **bound data** occupies — the design-preserving
/// plan gh#831 asks for.
///
/// `times` is the loader's own [`StreamTimes`] for the stream, `labels` the
/// values of its label column (the `: time` column, or `window_stop`), and
/// `cells` the loaded values, `None` for an `NA` hole. All three are parallel,
/// as `resolve_and_load_obs_streams` builds them.
///
/// `format` is how the bound file spelled its temporal cells, so the emitted
/// one spells them the same way (gh#882).
///
/// Nothing here is re-derived: the periods are the ones the fit will score
/// over, per-row window widths included, and a hole stays a row so the file
/// reloads with the same holes it was drawn on.
pub(crate) fn plan_bound_rows(
    obs: &ObservationModel,
    times: &StreamTimes,
    labels: &[f64],
    cells: &[Option<ObsCell>],
    format: TemporalFormat,
) -> Result<EmitPlan, String> {
    let columns = temporal_columns(obs)?;
    if times.len() != labels.len() || times.len() != cells.len() {
        return Err(format!(
            "observation stream '{}': the loader bound {} period(s), {} label(s) and \
             {} cell(s) — they must be one per row",
            obs.name, times.len(), labels.len(), cells.len()
        ));
    }
    if times.temporal_kind() != obs.projection.temporal_kind() {
        return Err(format!(
            "observation stream '{}': the bound rows read {:?} but the stream's \
             projection reads {:?}",
            obs.name, times.temporal_kind(), obs.projection.temporal_kind()
        ));
    }
    let rows = (0..times.len())
        .map(|k| EmitRow {
            label: labels[k],
            coverage: times.coverage(k),
            observed: cells[k].is_some(),
        })
        .collect();
    Ok(EmitPlan { columns, format, scored: obs.scored.clone(), rows })
}

/// One stream's rows in a dataset about to be written, and the identities the
/// writer and the fit config need: the file's stem is the stream's **name**,
/// and the key it binds to in `[data.observations]` is the stream's **source**.
#[derive(Debug, Clone)]
pub(crate) struct StreamPlan {
    pub name: String,
    pub source: String,
    pub plan: EmitPlan,
}

/// The design a fit's bound data fixes: per stream, the rows the loader gives
/// the fit (gh#831), under the temporal representation that fit's own files
/// used (gh#882).
///
/// `files` maps each stream's `source` to the data file bound to it — the
/// `effective` map the caller already resolved. The files are re-read here for
/// one thing only: whether their temporal cells were written as ISO dates, so
/// the emitted file states the boundaries the way the modeller published them
/// rather than as the day offsets they convert to. A model with no `origin`
/// has no date to render through and stays numeric.
///
/// The streams are checked first ([`check_streams_round_trip`]), so
/// `--design-from` refuses before it spends a forward simulation.
pub(crate) fn design_from_bound_streams(
    streams: &[crate::fit::runner::ObsStream],
    files: &indexmap::IndexMap<String, String>,
    time_opts: &crate::caltime_load::TimeOpts<'_>,
) -> Result<Vec<StreamPlan>, String> {
    let irs: Vec<&ObservationModel> = streams.iter().map(|s| &s.obs_model_ir).collect();
    check_streams_round_trip(&irs)?;
    streams.iter()
        .map(|s| {
            let obs = &s.obs_model_ir;
            let labels: Vec<f64> = s.data.iter().map(|o| o.time).collect();
            let format = bound_temporal_format(obs, files, time_opts)?;
            Ok(StreamPlan {
                name: obs.name.clone(),
                source: obs.source.clone(),
                plan: plan_bound_rows(obs, &s.times, &labels, &s.cells, format)?,
            })
        })
        .collect()
}

/// How the file bound to `obs` spelled its temporal cells. Dated only when the
/// file's cells were dates *and* the model carries the `origin` that renders
/// them — without an anchor there is no date to write, and the loader could
/// not have read one either.
fn bound_temporal_format(
    obs: &ObservationModel,
    files: &indexmap::IndexMap<String, String>,
    time_opts: &crate::caltime_load::TimeOpts<'_>,
) -> Result<TemporalFormat, String> {
    let (Some(path), Some(origin)) = (files.get(&obs.source), time_opts.origin) else {
        return Ok(TemporalFormat::Numeric);
    };
    match crate::pfilter::stream_cells_were_dated(obs, path, time_opts)? {
        true => Ok(TemporalFormat::Dated {
            origin: origin.to_string(),
            time_unit: time_opts.time_unit.to_string(),
        }),
        false => Ok(TemporalFormat::Numeric),
    }
}

/// Refuse, by name, a stream this writer cannot produce the loader's own file
/// for. Every simulated dataset is meant to be read back — that round trip is
/// what makes a recovery study a test of the model rather than of a
/// transcription — so a shape that would not re-load is a stop, not an output.
///
/// Three refusals:
///
/// - a stream whose likelihood reads a data column — a binomial denominator
///   `n = tested`, a person-time offset. There is no data file to read it from
///   when the data is what is being generated, and writing `0` would assert an
///   observation the run never made (gh#829): a synthetic file claiming zero
///   positives out of zero tests is scored as a real observation when it is
///   fitted back. The exception is the column [`model_denominator_column`]
///   names: a ratio stream's `n`, which the model generates.
/// - a stratified (long-form) stream. Its family's leaves share one `source`
///   and one long-form file with `: dim` columns; one file per leaf, with no
///   dim column, is a shape the loader would not route.
/// - two streams sharing one `source`. The loader binds one file per source, so
///   two files under one key cannot both be bound — one stream's data would go
///   unread.
fn check_streams_round_trip(streams: &[&ObservationModel]) -> Result<(), String> {
    for obs in streams {
        let aux = crate::pfilter::stream_aux_columns(obs);
        if !aux.is_empty() && model_denominator_column(obs).is_none() {
            return Err(format!(
                "observation stream '{}': its likelihood reads the data column(s) {} — \
                 values a data file supplies and the model has no term to generate. A \
                 simulated dataset has no file to read them from, and writing 0 would \
                 assert an observation the run never made (gh#829), which is then scored \
                 as real when the file is fitted back.\n  \
                 Fix: leave this stream out of the design (bind only the streams whose \
                 likelihood reads the model alone), or wait on gh#829, which lands the \
                 covariate-conditioned draw. (A `binomial` `n` over a ratio of flows is \
                 the one column the model does generate, and is written from it.)",
                obs.name,
                aux.iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", "),
            ));
        }
        if crate::pfilter::is_long_form_stream(obs) {
            return Err(format!(
                "observation stream '{}' is stratified: it declares `: dim` column(s), so \
                 its family shares one long-form file per `source` and the loader routes \
                 each row to a stratum leaf by name. This writer emits one file per \
                 stream, which that loader would not read back, so the stream is refused \
                 rather than written in a shape that does not re-load.",
                obs.name,
            ));
        }
    }
    for (i, a) in streams.iter().enumerate() {
        if let Some(b) = streams[..i].iter().find(|b| b.source == a.source) {
            return Err(format!(
                "observation streams '{}' and '{}' both read the source '{}', which the \
                 loader binds to ONE file — so the two files this would write cannot both \
                 be bound, and one stream's data would go unread. Give each stream its own \
                 `from` label, or fit them from a file you supply.",
                b.name, a.name, a.source,
            ));
        }
    }
    Ok(())
}

/// The one data column the emitter can write from the model: a `binomial` /
/// `beta_binomial` `n` that is exactly a declared column, on a stream whose
/// projection is a ratio of flows, and which is the only column the likelihood
/// reads. The ratio's denominator flow over the row is what `n` counts — the
/// events that were classified — so the model generates it, unlike a survey's
/// `tested`, which is surveillance effort the model has no term for (gh#829;
/// proposal 2026-09-09). Returns the column's name and the denominator's
/// transition names.
pub(crate) fn model_denominator_column(obs: &ObservationModel) -> Option<(String, Vec<String>)> {
    use ir::expr::Expr;
    use ir::observation::{Likelihood, Projection};
    let Projection::FlowRatio { denominator, .. } = &obs.projection else { return None };
    let n = match &obs.likelihood {
        Likelihood::Binomial(b) => &b.n,
        Likelihood::BetaBinomial(bb) => &bb.n,
        _ => return None,
    };
    let Expr::ObsColumnRef(w) = n else { return None };
    let col = w.obs_column_ref.clone();
    if crate::pfilter::stream_aux_columns(obs) != vec![col.clone()] {
        return None;
    }
    Some((col, denominator.clone()))
}

/// Where a simulated dataset's rows come from.
pub(crate) enum ObservationDesign<'a> {
    /// The rows a fit's bound data occupies, from [`design_from_bound_streams`]
    /// — every observed label, each row's own period, every `NA` hole, exactly
    /// as the loader bound them (gh#831).
    Bound(&'a [StreamPlan]),
    /// The rows the model's own `emit_schedule` declares — the design for a run
    /// with no data bound. Planned after the run, because the first window
    /// opens at the run's start. `emit` is the `--emit-every` override (gh#656).
    Declared { emit: Option<&'a crate::emit_every::EmitEvery> },
}

/// One stream's generated file.
#[derive(Debug, Clone)]
pub(crate) struct StreamFile {
    /// The `source` this file binds to in `[data.observations]` — the key a
    /// fit config uses to bind it back.
    pub source: String,
    /// The file, named after the stream.
    pub path: PathBuf,
}

/// Simulate one dataset and write it as the files the loader reads: one per
/// stream, under the stream's declared column names, into `out_dir`.
///
/// The forward run is `run` — the same [`crate::util::SimRun`] `simulate`
/// builds, so the parameter, scenario, backend and seed precedence is the one
/// path. Observation noise is drawn from `run.seed ^ SEED_MIX_OBS`, the
/// decorrelation constant every emitter shares, so the same nominal seed
/// produces the same observation bytes whichever verb asked for them.
///
/// Every stream is projected before any file is created: a projection that
/// cannot be read off the trajectory (a period boundary that is not a recorded
/// output time) then leaves no partial dataset behind.
pub(crate) fn simulate_dataset(
    run: &crate::util::SimRun,
    design: ObservationDesign<'_>,
    out_dir: &Path,
) -> Result<Vec<StreamFile>, String> {
    let (traj, model) = crate::util::run_simulation(run)?;
    if model.observations.is_empty() {
        return Err("model has no `observations { }` block — a simulated dataset \
             requires at least one observation stream in the .camdl file".to_string());
    }

    let plans: Vec<StreamPlan> = match design {
        ObservationDesign::Bound(plans) => plans.to_vec(),
        ObservationDesign::Declared { emit } => {
            // gh#656: refuse an override that names no stream, names a fit-only
            // stream, or targets an `at [...]` list — before any data is
            // written, so a mis-typed label never yields a silently unchanged
            // dataset.
            if let Some(e) = emit {
                e.validate(&model.observations)?;
            }
            // A bound design was checked when it was built, before this run;
            // a declared one is checked here, still before anything is written.
            check_streams_round_trip(&model.observations.iter().collect::<Vec<_>>())?;
            let run_start = crate::run_start_of(&traj, &model);
            model.observations.iter()
                .map(|obs| {
                    let times = crate::obs_emit_schedule_times(
                        obs, None, model.simulation.t_end, emit)?;
                    let plan = plan_emission(
                        obs, &times, run_start, model.simulation.t_end,
                        emit.and_then(|e| e.resolve_for(&obs.source)),
                    )?;
                    Ok(StreamPlan {
                        name: obs.name.clone(),
                        source: obs.source.clone(),
                        plan,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        }
    };

    // The stream's declaration as THIS run resolved it (a scenario can change
    // the model the trajectory came from), matched to the plan by name.
    let declaration = |name: &str| -> Result<&ObservationModel, String> {
        model.observations.iter().find(|o| o.name == name).ok_or_else(|| format!(
            "the observation design names stream '{name}', which the model this run \
             simulated does not declare"))
    };

    // Project every stream before creating any file: a partial dataset beside a
    // failure is worse than none (gh#589 review).
    let mut projected: Vec<Vec<f64>> = Vec::with_capacity(plans.len());
    // The rows' model denominators, for a ratio stream whose likelihood `n` is
    // a data column (proposal 2026-09-09): the denominator flow over each row,
    // rounded to the count it is, written under the declared column and handed
    // to the sampler as that row's aux. `None` for every other stream.
    let mut denominators: Vec<Option<(String, Vec<f64>)>> = Vec::with_capacity(plans.len());
    for sp in &plans {
        let obs = declaration(&sp.name)?;
        projected.push(crate::project_coverages(&traj, obs, &model, &sp.plan.coverages())?);
        denominators.push(match model_denominator_column(obs) {
            Some((col, flows)) => {
                let indices = crate::flow_indices_of(&model, &flows);
                let n = crate::incidence_over_rows(
                    &traj, &obs.name, &indices, &sp.plan.coverages(),
                )?;
                Some((col, n.into_iter().map(f64::round).collect()))
            }
            None => None,
        });
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("cannot create {}: {}", out_dir.display(), e))?;

    let compiled = Arc::new(
        CompiledModel::new(model.clone()).map_err(|e| format!("compile error: {:?}", e))?,
    );
    let params = compiled.default_params.clone();
    // One observation RNG, consumed in declaration order across streams — the
    // order every other emitter uses, so a shared seed means shared bytes.
    let mut obs_rng = StatefulRng::new(run.seed ^ crate::util::SEED_MIX_OBS);

    let mut written = Vec::with_capacity(plans.len());
    for ((sp, projected), denominator) in plans.iter().zip(projected).zip(denominators) {
        let obs = declaration(&sp.name)?;
        let sampler = sim::inference::obs_model::compile_obs_sample_pf(
            obs, compiled.clone(), &params,
        );
        // gh#6: the compartment state at each row's label, so likelihood arg
        // expressions (`p = projected / N`) resolve. The aux slice is the row's
        // model denominator for a ratio stream and empty otherwise — a stream
        // needing any other data column was refused above (gh#829). A row
        // whose denominator is 0 projects NaN and draws 0 of 0, which scores 0
        // when fitted back.
        let values: Vec<f64> = sp.plan.rows.iter().enumerate()
            .map(|(ti, row)| {
                let snap = crate::snap_at(&traj, row.label);
                let aux: Vec<(String, f64)> = denominator.as_ref()
                    .map(|(col, n)| vec![(col.clone(), n[ti])])
                    .unwrap_or_default();
                sampler(projected[ti], row.label, &snap.int_state.counts, &aux, &mut obs_rng)
            })
            .collect();
        let path = out_dir.join(format!("{}.tsv", sp.name));
        let extra: Vec<(String, Vec<f64>)> = denominator.into_iter().collect();
        write_stream_file(&path, &sp.plan, &values, &extra)?;
        written.push(StreamFile { source: sp.source.clone(), path });
    }
    Ok(written)
}

/// Write one stream's file: the declared temporal column(s), then the scored
/// column, then any `extra` columns, one line per planned row.
///
/// `values` is parallel to `plan.rows`. A row the plan marks unobserved is
/// written under the loader's hole token `NA`, keeping its period — the row
/// carries the reset the fit's accumulator needs even though it carries no
/// likelihood term. Each `extra` column is a declared data column the model
/// generated (a ratio stream's denominator, [`model_denominator_column`]),
/// parallel to the rows and written on hole rows too — the denominator is a
/// model quantity the row has whether or not its count was observed.
pub(crate) fn write_stream_file(
    path: &Path,
    plan: &EmitPlan,
    values: &[f64],
    extra: &[(String, Vec<f64>)],
) -> Result<(), String> {
    use std::io::Write;
    if values.len() != plan.rows.len() {
        return Err(format!(
            "{}: {} planned row(s) but {} value(s)",
            path.display(), plan.rows.len(), values.len()
        ));
    }
    for (name, column) in extra {
        if column.len() != plan.rows.len() {
            return Err(format!(
                "{}: {} planned row(s) but {} value(s) for column `{name}`",
                path.display(), plan.rows.len(), column.len()
            ));
        }
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {}", path.display(), e))?,
    );
    let io = |e: std::io::Error| format!("{}: {}", path.display(), e);
    match &plan.columns {
        TemporalColumns::Time(t) => write!(out, "{t}\t{}", plan.scored).map_err(io)?,
        TemporalColumns::Window { start, stop } => {
            write!(out, "{start}\t{stop}\t{}", plan.scored).map_err(io)?
        }
    }
    for (name, _) in extra {
        write!(out, "\t{name}").map_err(io)?;
    }
    writeln!(out).map_err(io)?;
    for (ri, (row, &value)) in plan.rows.iter().zip(values).enumerate() {
        let cell = |t: f64| plan.format.cell(t)
            .map_err(|e| format!("{}: {e}", path.display()));
        match (&plan.columns, row.coverage) {
            (TemporalColumns::Window { .. }, Coverage::Interval { start, stop }) => {
                write!(out, "{}\t{}", cell(start)?, cell(stop)?).map_err(io)?
            }
            _ => write!(out, "{}", cell(row.label)?).map_err(io)?,
        }
        match row.observed {
            true => write!(out, "\t{}", format_obs_value(value)).map_err(io)?,
            false => write!(out, "\tNA").map_err(io)?,
        }
        for (_, column) in extra {
            write!(out, "\t{}", format_obs_value(column[ri])).map_err(io)?;
        }
        writeln!(out).map_err(io)?;
    }
    out.flush().map_err(io)
}

/// A drawn observation as the file spells it: an integral value as an integer,
/// so a count file looks like a count file, and anything else at six decimals.
pub(crate) fn format_obs_value(v: f64) -> String {
    if v == v.round() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{:.6}", v)
    }
}

/// The `(column header, level)` cells a long-form row carries for one stratum
/// leaf: each declared `: dim` column, filled from this leaf's own `stratum`
/// (gh#884). Empty for an unstratified stream.
///
/// A stratified family shares one `source` and one long-format file, and the
/// loader routes each row to its leaf by these values — so a file written
/// without them cannot be read back at all, whatever else it carries.
pub(crate) fn dim_cells(obs: &ObservationModel) -> Result<Vec<(String, String)>, String> {
    obs.columns.iter()
        .filter_map(|c| match &c.role {
            ColumnRole::Dim(d) => Some((c.name.clone(), d.clone())),
            _ => None,
        })
        .map(|(header, dim)| {
            obs.stratum.iter()
                .find(|k| k.dim == dim)
                .map(|k| (header, k.level.clone()))
                .ok_or_else(|| format!(
                    "observation stream '{}' declares a `: dim` column for dimension \
                     '{dim}' but its own stratum names no level of it, so no long-form \
                     row could say which leaf it belongs to.",
                    obs.name))
        })
        .collect()
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

    // ── The design-preserving simulate (gh#831) ────────────────────────────
    //
    // The fixture is the committed `seed_timing` model with its stream's `:
    // time` column replaced by a `win_start`/`win_stop` pair, so its rows'
    // periods come from the data file rather than from a uniform rule. The
    // data file below is what no `covers` form can state: one-day rows, a
    // three-day row, a two-day row, and an `NA` hole.

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn path(&self) -> &Path { &self.0 }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir()
            .join(format!("camdl_design_{}_{}_{}", tag, std::process::id(), ns));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn seed_timing_ir() -> String {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        std::fs::read_to_string(
            Path::new(&manifest).join("../sim/tests/fixtures/seed_timing.ir.json"),
        ).unwrap()
    }

    /// The fixture's one stream re-declared with per-row window columns.
    fn windowed_model(dir: &Path) -> std::path::PathBuf {
        let mut v: serde_json::Value = serde_json::from_str(&seed_timing_ir()).unwrap();
        let obs = v["model"]["observations"][0].as_object_mut().expect("one stream");
        obs.insert("covers".into(), serde_json::json!({ "kind": "window_columns" }));
        let cols = obs["columns"].as_array_mut().expect("columns");
        let t = cols.iter().position(|c| c["role"] == "time").expect("a time column");
        cols.splice(t..=t, [
            serde_json::json!({ "name": "win_start", "role": "window_start" }),
            serde_json::json!({ "name": "win_stop",  "role": "window_stop" }),
        ]);
        let p = dir.join("windowed.ir.json");
        std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        p
    }

    /// The fixture's parameters, as the emitter integration test passes them.
    fn fixture_run(ir: &Path, seed: u64) -> crate::util::SimRun {
        let overrides: std::collections::HashMap<String, f64> = [
            ("beta", 0.6), ("gamma", 0.2), ("lambda", 2.0), ("w", 3.0),
            ("N0", 5000.0), ("rho", 0.5), ("k", 20.0), ("tau", 2.0),
        ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        crate::util::SimRun {
            ir_path: ir.to_string_lossy().into_owned(),
            overrides,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            seed,
            ..Default::default()
        }
    }

    /// The `source -> file` map a fit config's `[data.observations]` resolves
    /// to for these one-stream fixtures.
    fn effective(data_path: &Path) -> indexmap::IndexMap<String, String> {
        [("cases".to_string(), data_path.to_string_lossy().into_owned())]
            .into_iter().collect()
    }

    /// Load the streams a fit would bind for `data_path`, the way `fit run`
    /// binds them.
    fn bind(
        run: &crate::util::SimRun,
        data_path: &Path,
    ) -> Vec<crate::fit::runner::ObsStream> {
        let (compiled, model) = crate::util::resolve_run_model(run).unwrap();
        let opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt: run.dt,
            t_start: compiled.model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };
        crate::fit::runner::resolve_and_load_obs_streams(
            &model, &compiled, &effective(data_path), run.dt, &opts,
        ).unwrap()
    }

    /// The design `simulate --design-from` would preserve for `data_path`,
    /// built through the same seam that command uses.
    fn design_of(
        run: &crate::util::SimRun,
        data_path: &Path,
        bound: &[crate::fit::runner::ObsStream],
    ) -> Result<Vec<StreamPlan>, String> {
        let (compiled, model) = crate::util::resolve_run_model(run).unwrap();
        let opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt: run.dt,
            t_start: compiled.model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };
        design_from_bound_streams(bound, &effective(data_path), &opts)
    }

    /// gh#831. A dataset simulated on a fit's bound design re-loads with that
    /// design unchanged: every period the loader built — one-day rows, a
    /// three-day row, a two-day row — and every `NA` hole, which is a row with
    /// a period and no value rather than a row that is absent.
    ///
    /// This is the property the simulation-based self-consistency test rests
    /// on. A synthetic dataset on a regular grid would carry more information
    /// than the real one: the three-day row would become three one-day rows,
    /// and the hole would become an observation.
    #[test]
    fn a_dataset_simulated_on_a_bound_design_reloads_with_that_design() {
        let tmp = tempdir("roundtrip");
        let ir = windowed_model(tmp.path());
        let data = tmp.path().join("cases.tsv");
        std::fs::write(&data, "win_start\twin_stop\tcases\n\
             0\t1\t3\n\
             1\t2\t5\n\
             2\t5\t20\n\
             5\t6\t4\n\
             6\t7\tNA\n\
             7\t9\t11\n\
             9\t10\t6\n").unwrap();

        let run = fixture_run(&ir, 7);
        let bound = bind(&run, &data);
        assert_eq!(bound.len(), 1, "one stream");
        let want_times = bound[0].times.clone();
        let want_holes: Vec<bool> = bound[0].cells.iter().map(|c| c.is_none()).collect();
        // The design under test is irregular and has a hole — assert that,
        // so the round trip below cannot pass on a degenerate input.
        let widths: Vec<f64> = want_times.periods().unwrap().iter()
            .map(|p| p.width()).collect();
        assert_eq!(widths, vec![1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0]);
        assert_eq!(want_holes, vec![false, false, false, false, true, false, false]);

        let out = tmp.path().join("synth");
        let design = design_of(&run, &data, &bound).unwrap();
        let written = simulate_dataset(&run, ObservationDesign::Bound(&design), &out).unwrap();
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].source, "cases");

        let text = std::fs::read_to_string(&written[0].path).unwrap();
        assert_eq!(text.lines().next().unwrap(), "win_start\twin_stop\tcases",
            "the declared columns, so the loader reads the file back");
        assert!(text.lines().nth(5).unwrap().ends_with("\tNA"),
            "the hole is written as NA, keeping its period: {text}");

        let reloaded = bind(&run, &written[0].path);
        assert_eq!(reloaded[0].times, want_times,
            "every period the fit was bound over must survive the round trip");
        let got_holes: Vec<bool> = reloaded[0].cells.iter().map(|c| c.is_none()).collect();
        assert_eq!(got_holes, want_holes, "and every hole");
    }

    /// Proposal 2026-09-09: the one data column the model generates. A
    /// `binomial` `n` that is a declared column, over a ratio of flows, and the
    /// only column the likelihood reads, is the ratio's denominator; anything
    /// else that reads a data column stays a gh#829 refusal.
    #[test]
    fn the_model_denominator_is_a_binomial_n_over_a_ratio_of_flows_and_nothing_else() {
        use ir::expr::{BinOp, Expr, ProjectedExpr};
        use ir::observation::BinomialLikelihood;
        let ratio = || Projection::FlowRatio {
            numerator: vec!["die_comm".into()],
            denominator: vec!["die_comm".into(), "die_fac".into()],
        };
        let projected = || Expr::Projected(ProjectedExpr { projected: () });
        let mut cols = time_cols();
        cols.push(ObsColumn { name: "n_deaths".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) });
        let with = |projection: Projection, likelihood: Likelihood| {
            let mut s = stream(projection, Some(Covers::Until { offset: 0.0, span: 7.0 }), cols.clone());
            s.likelihood = likelihood;
            s
        };
        let binomial_n = || Likelihood::Binomial(BinomialLikelihood {
            n: Expr::obs_column_ref("n_deaths"),
            p: ir::Diffable::new(projected()),
        });
        assert_eq!(
            model_denominator_column(&with(ratio(), binomial_n())),
            Some(("n_deaths".to_string(), vec!["die_comm".to_string(), "die_fac".to_string()])),
            "a ratio's binomial n is the model's denominator"
        );
        assert_eq!(
            model_denominator_column(&with(incidence(), binomial_n())),
            None,
            "a count stream's n is surveillance effort, not a model quantity"
        );
        // The likelihood reads a second data column through `p`: not writable.
        let n_and_effort = Likelihood::Binomial(BinomialLikelihood {
            n: Expr::obs_column_ref("n_deaths"),
            p: ir::Diffable::new(Expr::bin_op(BinOp::Mul, projected(), Expr::obs_column_ref("effort"))),
        });
        assert_eq!(model_denominator_column(&with(ratio(), n_and_effort)), None);
        // `n` that is an expression over the column, not the column: not the
        // declared column's value, so not written.
        let n_expr = Likelihood::Binomial(BinomialLikelihood {
            n: Expr::bin_op(BinOp::Mul, Expr::obs_column_ref("n_deaths"), Expr::const_(2.0)),
            p: ir::Diffable::new(projected()),
        });
        assert_eq!(model_denominator_column(&with(ratio(), n_expr)), None);
    }

    /// gh#829. A stream whose likelihood reads a data column has no source for
    /// that column when the data is what is being generated. Writing 0 would
    /// assert an observation the run never made, so the design refuses the
    /// stream by name and names the column.
    #[test]
    fn a_covariate_stream_is_refused_by_name_rather_than_written_as_zeros() {
        let tmp = tempdir("covariate");
        // The fixture's negative-binomial mean `rho * projected` becomes
        // `tested * projected`, reading a declared data column.
        let mut v: serde_json::Value = serde_json::from_str(&seed_timing_ir()).unwrap();
        let obs = v["model"]["observations"][0].as_object_mut().unwrap();
        obs["columns"].as_array_mut().unwrap().push(
            serde_json::json!({ "name": "tested", "role": { "value": "count" } }));
        obs["likelihood"]["neg_binomial"]["mean"]["expr"]["bin_op"]["left"] =
            serde_json::json!({ "obs_column_ref": "tested" });
        let ir = tmp.path().join("covariate.ir.json");
        std::fs::write(&ir, serde_json::to_string_pretty(&v).unwrap()).unwrap();

        let data = tmp.path().join("cases.tsv");
        std::fs::write(&data, "time\tcases\ttested\n1\t3\t100\n2\t5\t120\n").unwrap();

        let run = fixture_run(&ir, 7);
        let bound = bind(&run, &data);
        let err = design_of(&run, &data, &bound)
            .expect_err("a covariate stream must be refused, not written as zeros");
        assert!(err.contains("'cases'") && err.contains("`tested`") && err.contains("gh#829"),
            "the refusal names the stream, the column and the issue: {err}");
    }

    // ── The window fraction against the instant ratio (proposal 2026-09-09) ──

    /// The golden `death_fraction_window` carries both spellings of the
    /// community-death fraction: `comm_frac`, the ratio of the two death flows
    /// accumulated over each week (a `FlowRatio`), and `comm_frac_instant`, the
    /// ratio of the two transitions' rates read at the week's closing label
    /// (`prevalence(...)`, the only spelling the compiler admits for it). On
    /// the deterministic backend the window stream must equal, to floating
    /// point, the ratio of the run's own `flow_die_comm` and `flow_die_fac`
    /// sums over each window; the instant stream is a different quantity, and
    /// sits below it because within each week the hospitalised pool is still
    /// rising relative to the infectious one. This is the proposal's table,
    /// asserted.
    #[test]
    fn the_window_fraction_is_the_runs_flow_ratio_and_the_instant_ratio_is_not() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir = Path::new(&manifest).join("../../../ocaml/golden/death_fraction_window.ir.json");
        let overrides: std::collections::HashMap<String, f64> = [
            ("beta", 0.5), ("gamma", 0.1), ("eta", 0.15), ("delta", 0.1),
            ("mu_c", 0.05), ("mu_f", 0.1), ("phi", 200.0), ("N0", 100000.0), ("I0", 10.0),
        ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let run = crate::util::SimRun {
            ir_path: ir.to_string_lossy().into_owned(),
            overrides,
            backend: crate::args::types::ForwardBackend::Ode,
            dt: 0.25,
            seed: 1,
            ..Default::default()
        };
        let (traj, model) = crate::util::run_simulation(&run).unwrap();
        let stream = |name: &str| model.observations.iter().find(|o| o.name == name).unwrap();

        let labels: Vec<f64> = (1..=12).map(|w| 7.0 * w as f64).collect();
        let windows: Vec<(f64, Coverage)> = labels.iter()
            .map(|&t| (t, Coverage::Interval { start: t - 7.0, stop: t })).collect();
        let instants: Vec<(f64, Coverage)> = labels.iter().map(|&t| (t, Coverage::Instant)).collect();
        let window = crate::project_coverages(&traj, stream("comm_frac"), &model, &windows).unwrap();
        let instant = crate::project_coverages(&traj, stream("comm_frac_instant"), &model, &instants).unwrap();

        // The reference, from the run's own per-snapshot flow columns: a
        // snapshot's flows are the events since the previous snapshot, so a
        // window's total is the sum over the snapshots in (start, stop].
        let tr = |n: &str| model.transitions.iter().position(|t| t.name == n).unwrap();
        let (die_comm, die_fac) = (tr("die_comm"), tr("die_fac"));
        let flow_over = |fi: usize, start: f64, stop: f64| -> f64 {
            traj.snapshots.iter()
                .filter(|s| s.t > start + 1e-9 && s.t <= stop + 1e-9)
                .map(|s| s.flows.value(fi))
                .sum()
        };
        let comp = |n: &str| model.compartments.iter().position(|c| c.name == n).unwrap();
        let (i_ix, h_ix) = (comp("I"), comp("H"));

        let mut table = String::from(
            "\n   t    Σ comm     Σ fac   window   instant@close   instant/window − 1\n");
        let mut gaps: Vec<f64> = Vec::new();
        for (k, &t) in labels.iter().enumerate() {
            let (c, f) = (flow_over(die_comm, t - 7.0, t), flow_over(die_fac, t - 7.0, t));
            let want = c / (c + f);
            assert!(
                (window[k] - want).abs() <= 1e-12,
                "week closing at {t}: the window stream must be Σ comm / (Σ comm + Σ fac) = \
                 {want} off the run's own flows, got {}", window[k]
            );
            // The instant stream reads the state at the label, off the same
            // snapshot the emitter reads (the rounded state on this backend).
            let snap = crate::snap_at(&traj, t);
            let (i, h) = (snap.int_state.counts[i_ix] as f64, snap.int_state.counts[h_ix] as f64);
            let want_instant = 0.05 * i / (0.05 * i + 0.1 * h);
            assert!(
                (instant[k] - want_instant).abs() <= 1e-12,
                "week closing at {t}: the instant stream is the rate ratio at the label, \
                 {want_instant}, got {}", instant[k]
            );
            let gap = instant[k] / want - 1.0;
            table.push_str(&format!(
                "{t:>4} {c:>9.2} {f:>9.2}   {want:.4}   {:.4}          {:+.3}\n", instant[k], gap
            ));
            gaps.push(gap);
        }
        eprintln!("window fraction vs instant ratio, weekly windows on the ODE backend:{table}");

        // The divergence. Every week's instant ratio sits below the window
        // fraction, and at the worst week by more than a tenth: a modeller
        // fitting a community share against the instant spelling is pulled
        // low, most where the flows move fastest.
        assert!(gaps.iter().all(|&g| g < 0.0),
            "the instant ratio sits below the window fraction every week: {gaps:?}");
        let worst = gaps.iter().cloned().fold(0.0_f64, f64::min);
        assert!(worst < -0.10, "the worst week's gap exceeds a tenth: {worst:.3} ({gaps:?})");
    }
}
