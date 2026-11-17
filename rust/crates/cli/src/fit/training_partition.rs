//! Which rows the parameters were fitted to (`docs/dev/temporal-architecture.md`
//! §2.3, §3.2, §4.1, §4.7; D38, D40).
//!
//! A [`TrainingPartition`] is decided once, from the rows the one row reader
//! ([`super::observed_rows::observed_rows`]) produced, and recorded in the
//! fit's `fit.meta.json`. Per stream it holds a [`RowPrefix`]: how many of the
//! stream's leading rows trained and the row digest of exactly those rows.
//! Under `holdout_after` a row trains if the end of what it measures is at or
//! before the cutoff; under `[data.holdout]`, or with no holdout, every row of
//! the training files trains. The **training extent** is the latest end among
//! the training rows of every stream.
//!
//! Membership is "the first `rows` rows", so no consumer compares a time with a
//! cutoff again, and the digest proves the fitted rows are unchanged when a
//! later command reads the file again.

use std::collections::BTreeMap;

use runid::{CanonicalHasher, ContentHash, FiniteF64};

use super::observed_rows::ObservedRow;
use crate::caltime_load::TemporalCell;

/// The version of the rule that decides which rows train. Recorded with every
/// partition and folded into the fit identity
/// (`runid::inputs::FitDigest::partition_rule_version`), so a change to the
/// rule bumps it and re-keys every fit with no `HASH_VERSION` bump.
pub(crate) const PARTITION_RULE_VERSION: u16 = 1;

/// The version of the row digest's encoding (the score-artifact proposal's row
/// digest v1, `docs/dev/proposals/2026-11-14-score-artifact.md`, "The row
/// digest").
pub(crate) const ROW_DIGEST_VERSION: u16 = 1;

/// "The same observation time" at the training boundary: float noise from a
/// date conversion must not move a row across the cut.
use super::runner::TIME_TIE_EPS;

/// The row digest of a run of one stream's rows: "are these the same
/// observations", independent of how the file lays them out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowDigest(ContentHash);

impl RowDigest {
    pub(crate) fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Row digest version 1 over `rows`, in order.
///
/// Hashes the digest version; the number of temporal columns the stream binds
/// (2 for per-row windows, 1 otherwise) and each column's kind (0 numbers, 1
/// dates); the row count; and per row its temporal cells as written (a number
/// by value, so `7`, `7.0` and `07` agree; a date as its text), its cell (a
/// `u8` tag, 0 for `NA` and 1 for a value followed by the value) and its
/// auxiliary values sorted by name. The leaf's name is not hashed: one series
/// has one digest wherever it is bound.
pub(crate) fn row_digest(rows: &[ObservedRow]) -> Result<RowDigest, String> {
    let finite = |x: f64, what: &str| FiniteF64::new(x)
        .map_err(|_| format!("internal error: a non-finite {what} ({x}) reached the row digest"));
    let mut h = CanonicalHasher::new();
    h.write_u16(ROW_DIGEST_VERSION);
    // Every row of a stream has the same temporal columns, each of one kind
    // (the loader detects a column's kind over all its cells).
    let columns: &[TemporalCell] = rows.first().map(|r| r.temporal.as_slice()).unwrap_or(&[]);
    let n_columns = if columns.is_empty() { 1 } else { columns.len() };
    h.write_u8(n_columns as u8);
    for i in 0..n_columns {
        h.write_u8(match columns.get(i) {
            Some(TemporalCell::Date(_)) => 1,
            Some(TemporalCell::Number(_)) | None => 0,
        });
    }
    h.write_u64(rows.len() as u64);
    for row in rows {
        for cell in &row.temporal {
            match cell {
                TemporalCell::Number(x) => {
                    h.write_f64_bits(finite(*x, "time cell")?.get());
                }
                TemporalCell::Date(d) => h.write_str(d),
            }
        }
        match row.cell {
            None => h.write_u8(0),
            Some(sim::inference::ObsCell::Scalar(v)) => {
                h.write_u8(1);
                h.write_f64_bits(finite(v, "observed value")?.get());
            }
        }
        let mut aux: Vec<&(String, f64)> = row.aux.iter().collect();
        aux.sort_by(|a, b| a.0.cmp(&b.0));
        h.write_u64(aux.len() as u64);
        for (name, v) in aux {
            h.write_str(name);
            h.write_f64_bits(finite(*v, "auxiliary value")?.get());
        }
    }
    Ok(RowDigest(h.finalize()))
}

/// How a partition was decided: provenance, recorded beside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PartitionRule {
    /// No holdout declared: every row trains.
    All,
    /// `holdout_after`: a row trains if its end is at or before `at`.
    Cutoff { at: f64 },
    /// `[data.holdout]`: every row of the training files trains; the held-out
    /// rows are in separate files.
    SeparateFiles,
}

/// Which whole-file rule [`TrainingPartition::whole`] records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WholeFiles {
    /// No holdout declared.
    NoHoldout,
    /// `[data.holdout]` names the held-out rows' files.
    SeparateHoldoutFiles,
}

/// One stream's training rows: rows `[0, rows)` of the stream, and their
/// digest.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowPrefix {
    pub stream: String,
    pub rows: usize,
    pub digest: RowDigest,
}

/// Why a partition cannot be built.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PartitionError {
    /// A stream has no row with an observed value at or before the cutoff.
    NoTrainingRows { stream: String },
    /// No row with an observed value ends after the cutoff, in any stream:
    /// nothing is held out.
    NothingHeldOut,
    /// No stream has any row, so there is no extent.
    NoRows,
    /// The rows that train are not a prefix of the stream: a row that trains
    /// follows one that does not. Unreachable while the binder requires each
    /// stream's ends to increase; checked rather than assumed.
    NotAPrefix { stream: String, position: usize },
    /// A value the digest cannot hash.
    Digest(String),
}

/// Whether `row` trains under `holdout_after = at`: the end of what it
/// measures is at or before `at` (within [`TIME_TIE_EPS`]). The one statement
/// of the cutoff rule; [`cutoff_prefix`] applies it to a stream.
pub(crate) fn trains_under_cutoff(row: &ObservedRow, at: f64) -> bool {
    row.measures.end() <= at + TIME_TIE_EPS
}

/// How many leading rows of one stream train under `holdout_after = at`
/// ([`trains_under_cutoff`]), or `Err(position)` when the row at `position`
/// trains after one that does not, so the training rows are not a prefix.
/// [`TrainingPartition::from_cutoff`] and `pfilter --save-final-state`
/// (gh#1138) both cut a stream through it.
pub(crate) fn cutoff_prefix(rows: &[ObservedRow], at: f64) -> Result<usize, usize> {
    let n = rows.iter().take_while(|r| trains_under_cutoff(r, at)).count();
    match rows[n..].iter().position(|r| trains_under_cutoff(r, at)) {
        Some(position) => Err(n + position),
        None => Ok(n),
    }
}

/// Which rows of each stream the parameters were fitted to (spec §3.2).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TrainingPartition {
    rule: PartitionRule,
    /// The latest end among the training rows of every stream.
    extent: f64,
    /// Sorted by stream name.
    streams: Vec<RowPrefix>,
}

impl TrainingPartition {
    /// The partition `holdout_after = at` makes: in each stream the rows whose
    /// end is at or before `at` (within [`TIME_TIE_EPS`]). Every stream must
    /// keep a training row with an observed value, and some row with an
    /// observed value must be held out. An `NA` row is still a row: it keeps
    /// its position in the prefix and is hashed into the digest, but it is no
    /// observation to train on or to score (D40).
    pub(crate) fn from_cutoff(
        rows: &BTreeMap<String, Vec<ObservedRow>>,
        at: f64,
    ) -> Result<Self, PartitionError> {
        let observed = |rs: &[ObservedRow]| rs.iter().filter(|r| r.cell.is_some()).count();
        let mut streams = Vec::with_capacity(rows.len());
        let mut held_out = 0usize;
        for (stream, rs) in rows {
            let n = cutoff_prefix(rs, at).map_err(|position| PartitionError::NotAPrefix {
                stream: stream.clone(), position,
            })?;
            if observed(&rs[..n]) == 0 {
                return Err(PartitionError::NoTrainingRows { stream: stream.clone() });
            }
            held_out += observed(&rs[n..]);
            streams.push(prefix(stream, &rs[..n])?);
        }
        if held_out == 0 {
            return Err(PartitionError::NothingHeldOut);
        }
        Self::new(PartitionRule::Cutoff { at }, rows, streams)
    }

    /// The partition in which every row of every stream trains: no holdout, or
    /// `[data.holdout]` with its held-out rows in separate files.
    pub(crate) fn whole(
        rows: &BTreeMap<String, Vec<ObservedRow>>,
        files: WholeFiles,
    ) -> Result<Self, PartitionError> {
        let streams = rows.iter()
            .map(|(stream, rs)| prefix(stream, rs))
            .collect::<Result<Vec<_>, _>>()?;
        let rule = match files {
            WholeFiles::NoHoldout => PartitionRule::All,
            WholeFiles::SeparateHoldoutFiles => PartitionRule::SeparateFiles,
        };
        Self::new(rule, rows, streams)
    }

    fn new(
        rule: PartitionRule,
        rows: &BTreeMap<String, Vec<ObservedRow>>,
        streams: Vec<RowPrefix>,
    ) -> Result<Self, PartitionError> {
        let extent = streams.iter()
            .flat_map(|p| rows[&p.stream][..p.rows].iter().map(|r| r.measures.end()))
            .reduce(f64::max)
            .ok_or(PartitionError::NoRows)?;
        Ok(TrainingPartition { rule, extent, streams })
    }

    /// The partition a fit recorded, checked against the rows read now: every
    /// stream of `rows` must have a recorded prefix, its file must still hold
    /// at least that many rows, and those rows must digest to the recorded
    /// digest. A forecast reads its parameter set only through this, so a plan
    /// can never claim the parameters were fitted to rows that changed after
    /// the fit (`docs/dev/temporal-architecture.md` §3.3).
    pub(crate) fn from_record(
        record: &crate::run_meta::TrainingPartitionRecord,
        rows: &BTreeMap<String, Vec<ObservedRow>>,
    ) -> Result<Self, String> {
        use crate::run_meta::PartitionRuleRecord;
        if record.rule_version != PARTITION_RULE_VERSION {
            return Err(format!(
                "the fit recorded its training rows under partition rule version {}, \
                 but this camdl decides them under version {PARTITION_RULE_VERSION}; \
                 re-run the fit so its training rows are recorded by the current rule.",
                record.rule_version));
        }
        let mut streams = Vec::with_capacity(rows.len());
        for (stream, rs) in rows {
            let Some(p) = record.prefix(stream) else {
                return Err(format!(
                    "the fit's training partition records no rows for stream '{stream}', \
                     so which of its rows trained cannot be checked; re-run the fit."));
            };
            if rs.len() < p.rows {
                return Err(format!(
                    "the fit trained on the first {} row(s) of stream '{stream}', but its \
                     data now holds {} row(s): the data changed after the fit. Re-run the \
                     fit on the current data.", p.rows, rs.len()));
            }
            let digest = row_digest(&rs[..p.rows])?;
            if digest.to_hex() != p.digest {
                return Err(format!(
                    "the first {} row(s) of stream '{stream}' are not the rows the fit \
                     trained on (their row digest differs): the data changed after the \
                     fit, so its parameters were not fitted to them. Re-run the fit on \
                     the current data.", p.rows));
            }
            streams.push(RowPrefix { stream: stream.clone(), rows: p.rows, digest });
        }
        let rule = match &record.rule {
            PartitionRuleRecord::All => PartitionRule::All,
            PartitionRuleRecord::Cutoff { at } => PartitionRule::Cutoff { at: at.t },
            PartitionRuleRecord::SeparateFiles => PartitionRule::SeparateFiles,
        };
        let partition = Self::new(rule, rows, streams).map_err(|e| super::runner::partition_error(e, None))?;
        if (partition.extent - record.extent.t).abs() > TIME_TIE_EPS {
            return Err(format!(
                "the fit recorded its training extent as t = {}, but its recorded \
                 training rows end at t = {}; the record is inconsistent. Re-run the fit.",
                record.extent.t, partition.extent));
        }
        Ok(partition)
    }

    pub(crate) fn rule(&self) -> PartitionRule {
        self.rule
    }

    /// The latest end among the training rows of every stream.
    pub(crate) fn extent(&self) -> f64 {
        self.extent
    }

    /// Each stream's training prefix, sorted by stream name.
    pub(crate) fn streams(&self) -> &[RowPrefix] {
        &self.streams
    }

    /// The number of `stream`'s leading rows that trained; 0 for a stream the
    /// partition does not hold.
    pub(crate) fn rows(&self, stream: &str) -> usize {
        self.streams.iter().find(|p| p.stream == stream).map_or(0, |p| p.rows)
    }

    /// What the fit records in `fit.meta.json`, with each time also as a date
    /// when the model has an `origin`, written by the model's `calendar` axis
    /// (the one renderer of the `calendar` block the same sidecar stores).
    pub(crate) fn record(&self, calendar: &ir::caltime::CalendarAxis)
        -> crate::run_meta::TrainingPartitionRecord
    {
        use crate::run_meta::{PartitionRuleRecord, RowPrefixRecord, TimePoint};
        let point = |t: f64| TimePoint {
            t,
            date: calendar.date_of(t).map(ir::caltime::RenderedInstant::into_string),
        };
        crate::run_meta::TrainingPartitionRecord {
            rule_version: PARTITION_RULE_VERSION,
            rule: match self.rule {
                PartitionRule::All => PartitionRuleRecord::All,
                PartitionRule::Cutoff { at } => PartitionRuleRecord::Cutoff { at: point(at) },
                PartitionRule::SeparateFiles => PartitionRuleRecord::SeparateFiles,
            },
            extent: point(self.extent),
            streams: self.streams.iter()
                .map(|p| RowPrefixRecord {
                    stream: p.stream.clone(), rows: p.rows, digest: p.digest.to_hex(),
                })
                .collect(),
        }
    }
}

fn prefix(stream: &str, rows: &[ObservedRow]) -> Result<RowPrefix, PartitionError> {
    Ok(RowPrefix {
        stream: stream.to_string(),
        rows: rows.len(),
        digest: row_digest(rows).map_err(PartitionError::Digest)?,
    })
}

/// The observed window of `rows`: the earliest start and the latest end of a
/// row that carries a value (D40). An `NA` row was not observed and counts
/// toward neither. `None` when no row carries a value.
pub(crate) fn observed_window<'a>(rows: impl IntoIterator<Item = &'a ObservedRow>)
    -> Option<(f64, f64)>
{
    rows.into_iter()
        .filter(|r| r.cell.is_some())
        .map(|r| (r.measures.start(), r.measures.end()))
        .reduce(|(a, b), (c, d)| (a.min(c), b.max(d)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caltime_load::{TimeFormat, TimeOpts};
    use ir::observation::*;

    fn opts() -> TimeOpts<'static> {
        TimeOpts { origin: None, time_unit: "days", dt: 1.0, t_start: 0.0, format: TimeFormat::Auto }
    }

    fn stream(covers: Covers) -> ObservationModel {
        ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn { name: "cases".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "cases".into(),
            emit_schedule: None,
            stratum: vec![],
            covers: Some(covers),
            projection: Projection::CumulativeFlow("infection".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
            }),
        }
    }

    fn rows_of(name: &str, covers: Covers, body: &str) -> Vec<ObservedRow> {
        let dir = crate::test_support::unique_temp_dir(&format!("training_partition_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.tsv");
        std::fs::write(&path, body).unwrap();
        let obs = stream(covers);
        super::super::observed_rows::observed_rows(path.to_str().unwrap(), &obs, &[&obs], &opts())
            .unwrap()
    }

    fn one(rows: Vec<ObservedRow>) -> BTreeMap<String, Vec<ObservedRow>> {
        BTreeMap::from([("cases".to_string(), rows)])
    }

    const DAY: Covers = Covers::From { offset: 0.0, span: 1.0 };
    const CLOSING_7: Covers = Covers::Until { offset: 0.0, span: 7.0 };

    /// A `day(time)` row labelled D ends at D + 1, so a cutoff at 10 trains
    /// rows 1–9 and the extent is 10.
    #[test]
    fn a_day_row_trains_when_its_day_has_ended_by_the_cutoff() {
        let body: String = std::iter::once("time\tcases\n".to_string())
            .chain((1..=16).map(|d| format!("{d}\t{d}\n"))).collect();
        let rows = one(rows_of("day", DAY, &body));
        let p = TrainingPartition::from_cutoff(&rows, 10.0).unwrap();
        assert_eq!(p.rows("cases"), 9);
        assert_eq!(p.extent(), 10.0);
        assert_eq!(p.streams()[0].rows, 9);
        assert_eq!(p.rule(), PartitionRule::Cutoff { at: 10.0 });
    }

    /// The tolerance keeps a row whose end is a few ulps past the cutoff.
    #[test]
    fn an_end_at_the_cutoff_within_float_noise_trains() {
        let rows = one(rows_of("tie", CLOSING_7, "time\tcases\n7\t1\n14\t2\n21\t3\n"));
        let p = TrainingPartition::from_cutoff(&rows, 14.0 - 1e-12).unwrap();
        assert_eq!(p.rows("cases"), 2);
    }

    #[test]
    fn a_cutoff_before_every_row_and_after_every_row_is_refused() {
        let rows = one(rows_of("edges", CLOSING_7, "time\tcases\n7\t1\n14\t2\n"));
        assert_eq!(TrainingPartition::from_cutoff(&rows, 6.0),
            Err(PartitionError::NoTrainingRows { stream: "cases".into() }));
        assert_eq!(TrainingPartition::from_cutoff(&rows, 14.0),
            Err(PartitionError::NothingHeldOut));
    }

    /// A stream whose rows at or before the cutoff are all `NA` keeps no
    /// observed value to train on, so it is refused like a stream with no row
    /// there at all.
    #[test]
    fn a_stream_whose_training_rows_are_all_holes_is_refused() {
        let rows = one(rows_of("holes_train", DAY, "time\tcases\n1\tNA\n2\tNA\n20\t5\n21\t6\n"));
        assert_eq!(TrainingPartition::from_cutoff(&rows, 10.0),
            Err(PartitionError::NoTrainingRows { stream: "cases".into() }));
    }

    /// A cutoff whose only held-out row is `NA` holds out nothing `compare`
    /// could score, so it is refused as nothing held out.
    #[test]
    fn a_cutoff_whose_only_held_out_row_is_a_hole_holds_out_nothing() {
        let rows = one(rows_of("holes_held", DAY, "time\tcases\n1\t1\n2\t2\n20\tNA\n"));
        assert_eq!(TrainingPartition::from_cutoff(&rows, 10.0),
            Err(PartitionError::NothingHeldOut));
    }

    /// The holes still count as rows: positions and the digest cover them.
    #[test]
    fn holes_keep_their_positions_in_the_prefix() {
        let rows = one(rows_of("holes_pos", DAY, "time\tcases\n1\tNA\n2\t3\n20\t5\n"));
        let p = TrainingPartition::from_cutoff(&rows, 10.0).unwrap();
        assert_eq!(p.rows("cases"), 2, "the leading hole is row 0 of the prefix");
        assert_eq!(p.streams()[0].digest, row_digest(&rows["cases"][..2]).unwrap());
    }

    /// `whole` trains every row; its extent is the latest end, an `NA` row
    /// included, since the hole's period was part of the fit.
    #[test]
    fn whole_trains_every_row() {
        let rows = one(rows_of("whole", CLOSING_7, "time\tcases\n7\t1\n14\tNA\n"));
        let p = TrainingPartition::whole(&rows, WholeFiles::SeparateHoldoutFiles).unwrap();
        assert_eq!(p.rows("cases"), 2);
        assert_eq!(p.extent(), 14.0);
        assert_eq!(p.rule(), PartitionRule::SeparateFiles);
    }

    /// The digest hashes a numeric time cell by value and covers exactly the
    /// prefix: appending rows after it leaves it alone, editing a row in it
    /// changes it, and an `NA` differs from a value.
    #[test]
    fn the_row_digest_reads_the_cells_by_value() {
        let a = rows_of("d_a", CLOSING_7, "time\tcases\n7\t1\n14\t2\n");
        let b = rows_of("d_b", CLOSING_7, "time\tcases\n7.0\t1\n014\t2.0\n21\t5\n");
        let c = rows_of("d_c", CLOSING_7, "time\tcases\n7\t1\n14\t3\n");
        let na = rows_of("d_na", CLOSING_7, "time\tcases\n7\t1\n14\tNA\n");
        let d = |r: &[ObservedRow]| row_digest(r).unwrap();
        assert_eq!(d(&a), d(&b[..2]), "7, 7.0 and 07 hash alike; a later row is not hashed");
        assert_ne!(d(&a), d(&c), "an edited value changes the digest");
        assert_ne!(d(&a), d(&na), "a hole is not a value");
        assert_ne!(d(&a), d(&b), "the row count is hashed");
    }

    /// The record dates each time through the model's calendar axis, as the
    /// sidecar's `calendar` block would; no dates without an origin.
    #[test]
    fn the_record_dates_times_through_the_calendar_axis() {
        let rows = one(rows_of("record", CLOSING_7, "time\tcases\n7\t1\n14\t2\n21\t3\n"));
        let p = TrainingPartition::from_cutoff(&rows, 14.5).unwrap();
        let axis = ir::caltime::CalendarAxis::parse("days", Some("2020-01-01")).unwrap();
        let r = p.record(&axis);
        assert_eq!(r.extent.date.as_deref(), Some("2020-01-15"));
        let crate::run_meta::PartitionRuleRecord::Cutoff { at } = &r.rule else {
            panic!("a cutoff rule: {:?}", r.rule);
        };
        assert_eq!(at.date, axis.date_of(14.5).map(|d| d.into_string()));
        let numeric = ir::caltime::CalendarAxis::parse("days", None).unwrap();
        assert_eq!(p.record(&numeric).extent.date, None);
    }

    /// The observed window leaves out `NA` rows at either end.
    #[test]
    fn the_observed_window_skips_unobserved_rows() {
        let rows = rows_of("window", DAY, "time\tcases\n1\tNA\n2\t3\n3\t4\n4\tNA\n");
        assert_eq!(observed_window(&rows), Some((2.0, 4.0)));
    }
}
