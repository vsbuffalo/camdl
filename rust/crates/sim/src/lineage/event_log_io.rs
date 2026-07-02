//! On-disk I/O for the Layer-1 [`EventLog`] (the `simulate --event-log`
//! artifact and the `lineage realize` input).
//!
//! Two formats, mirroring the line-list writer conventions:
//!
//! - **TSV** ([`write_tsv`] / [`read_tsv`]) — dependency-free, human-inspectable.
//!   Two metadata header lines carry the (small) initial-pool seeding and the
//!   per-transition route table as JSON; the body is one tab-separated row per
//!   event. The `lineage_weights` cell is a JSON array (the recorded `w_b·X_b`
//!   masses) at lineage events, `-` otherwise.
//! - **Parquet** ([`write_parquet`] / [`read_parquet`]) — columnar event body,
//!   with `initial_pools` and `transitions` stored as JSON in the file's
//!   key-value metadata. Behind the `lineage-parquet` feature.
//!
//! The event log is identity-free and self-contained: a reader reconstructs the
//! full [`EventLog`] from the file alone, with no model.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::SimError;

use super::event_log::{EventLog, EventRecord, RouteInfo};
use super::writer::LineListFormat;
use super::{CompartmentId, DemeId, TransitionId};

/// Event-log metadata: the t=0 tracked pools and the per-transition route table.
/// Read up front (Parquet footer / TSV header) so events can then be streamed.
type EventLogMeta = (Vec<(DemeId, CompartmentId, i64)>, Vec<RouteInfo>);

/// On-disk format tag (TSV reuses [`LineListFormat`]).
const TSV_MAGIC: &str = "# camdl-event-log v1";

fn weights_cell(w: &Option<Vec<f64>>) -> String {
    match w {
        Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string()),
        None => "-".to_string(),
    }
}

fn parse_weights_cell(s: &str) -> Result<Option<Vec<f64>>, SimError> {
    if s == "-" {
        return Ok(None);
    }
    serde_json::from_str::<Vec<f64>>(s)
        .map(Some)
        .map_err(|e| SimError::Validation(format!("event log: bad lineage_weights cell '{}': {}", s, e)))
}

const EVENT_COLUMNS: &[&str] =
    &["time", "transition", "multiplicity", "batched", "step", "lineage_weights"];

/// Parse one TSV data row's tab-split fields into an [`EventRecord`].
/// `row_for_error` is the 1-based file line number, for diagnostics only.
fn parse_event_fields(f: &[&str], row_for_error: usize) -> Result<EventRecord, SimError> {
    if f.len() != EVENT_COLUMNS.len() {
        return Err(SimError::Validation(format!(
            "event log row {}: expected {} columns, got {}",
            row_for_error,
            EVENT_COLUMNS.len(),
            f.len()
        )));
    }
    let time: f64 = f[0]
        .parse()
        .map_err(|e| SimError::Validation(format!("event log time '{}': {}", f[0], e)))?;
    let transition: usize = f[1]
        .parse()
        .map_err(|e| SimError::Validation(format!("event log transition '{}': {}", f[1], e)))?;
    let multiplicity: u64 = f[2]
        .parse()
        .map_err(|e| SimError::Validation(format!("event log multiplicity '{}': {}", f[2], e)))?;
    let batched: bool = f[3]
        .parse()
        .map_err(|e| SimError::Validation(format!("event log batched '{}': {}", f[3], e)))?;
    let step: u64 = f[4]
        .parse()
        .map_err(|e| SimError::Validation(format!("event log step '{}': {}", f[4], e)))?;
    let lineage_weights = parse_weights_cell(f[5])?;
    Ok(EventRecord {
        time,
        transition: TransitionId(transition),
        multiplicity,
        batched,
        step,
        lineage_weights,
    })
}

/// Parse one TSV metadata header line (`# initial_pools\t…` / `# transitions\t…`)
/// into the metadata accumulators.
fn parse_tsv_meta_line(
    line: &str,
    initial_pools: &mut Option<Vec<(DemeId, CompartmentId, i64)>>,
    transitions: &mut Option<Vec<RouteInfo>>,
) -> Result<(), SimError> {
    if let Some(rest) = line.strip_prefix("# initial_pools\t") {
        *initial_pools = Some(
            serde_json::from_str(rest)
                .map_err(|e| SimError::Validation(format!("event log initial_pools: {}", e)))?,
        );
    } else if let Some(rest) = line.strip_prefix("# transitions\t") {
        *transitions = Some(
            serde_json::from_str(rest)
                .map_err(|e| SimError::Validation(format!("event log transitions: {}", e)))?,
        );
    } else {
        return Err(SimError::Validation(format!(
            "event log: unexpected metadata line '{}'",
            line
        )));
    }
    Ok(())
}

/// Write the event log as TSV to a path.
pub fn write_tsv(log: &EventLog, path: &Path) -> Result<(), SimError> {
    let file = File::create(path).map_err(|e| {
        SimError::Validation(format!("cannot create event log {}: {}", path.display(), e))
    })?;
    let mut out = BufWriter::new(file);
    write_tsv_into(log, &mut out)?;
    out.flush()
        .map_err(|e| SimError::Validation(format!("event log flush: {}", e)))
}

/// Serialize the event log as TSV into any writer. The single source of the
/// canonical TSV byte layout — shared by [`write_tsv`] (to a file) and
/// [`to_tsv_bytes`] (to the in-leaf `event_log.tsv` CAS artifact), so the
/// stored artifact and the `--event-log PATH` mirror are byte-identical.
pub fn write_tsv_into<W: Write>(log: &EventLog, out: &mut W) -> Result<(), SimError> {
    let werr = |e: std::io::Error| SimError::Validation(format!("event log write: {}", e));
    writeln!(out, "{}", TSV_MAGIC).map_err(werr)?;
    let pools_json = serde_json::to_string(&log.initial_pools)
        .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
    let routes_json = serde_json::to_string(&log.transitions)
        .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
    writeln!(out, "# initial_pools\t{}", pools_json).map_err(werr)?;
    writeln!(out, "# transitions\t{}", routes_json).map_err(werr)?;
    writeln!(out, "{}", EVENT_COLUMNS.join("\t")).map_err(werr)?;
    for e in &log.events {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}",
            e.time,
            e.transition.0,
            e.multiplicity,
            e.batched,
            e.step,
            weights_cell(&e.lineage_weights),
        )
        .map_err(werr)?;
    }
    Ok(())
}

/// The canonical TSV bytes of an event log, for content-addressed storage as
/// the `event_log.tsv` artifact alongside `traj.tsv` in a sim leaf.
pub fn to_tsv_bytes(log: &EventLog) -> Result<Vec<u8>, SimError> {
    let mut buf = Vec::new();
    write_tsv_into(log, &mut buf)?;
    Ok(buf)
}

/// Read a TSV event log back into an [`EventLog`].
pub fn read_tsv(path: &Path) -> Result<EventLog, SimError> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| SimError::Validation(format!("read event log {}: {}", path.display(), e)))?;
    let mut lines = body.lines();
    let magic = lines
        .next()
        .ok_or_else(|| SimError::Validation("empty event log".to_string()))?;
    if magic != TSV_MAGIC {
        return Err(SimError::Validation(format!(
            "event log: bad magic line '{}', expected '{}'",
            magic, TSV_MAGIC
        )));
    }

    let mut initial_pools = None;
    let mut transitions = None;
    // The two metadata header lines, then the column header.
    for _ in 0..2 {
        let line = lines
            .next()
            .ok_or_else(|| SimError::Validation("event log: truncated metadata".to_string()))?;
        parse_tsv_meta_line(line, &mut initial_pools, &mut transitions)?;
    }
    let header = lines
        .next()
        .ok_or_else(|| SimError::Validation("event log: missing column header".to_string()))?;
    if header != EVENT_COLUMNS.join("\t") {
        return Err(SimError::Validation(format!(
            "event log header mismatch: expected '{}', got '{}'",
            EVENT_COLUMNS.join("\t"),
            header
        )));
    }

    let mut events = Vec::new();
    for (lineno, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        events.push(parse_event_fields(&f, lineno + 5)?);
    }

    Ok(EventLog {
        initial_pools: initial_pools
            .ok_or_else(|| SimError::Validation("event log: missing initial_pools".to_string()))?,
        transitions: transitions
            .ok_or_else(|| SimError::Validation("event log: missing transitions".to_string()))?,
        events,
    })
}

/// TSV header reader: magic + the two metadata lines, stopping before the rows.
/// Cheap regardless of file size (reads ~4 lines).
fn read_metadata_tsv(path: &Path) -> Result<EventLogMeta, SimError> {
    let file = File::open(path)
        .map_err(|e| SimError::Validation(format!("read event log {}: {}", path.display(), e)))?;
    let mut lines = BufReader::new(file).lines();
    let magic = lines
        .next()
        .ok_or_else(|| SimError::Validation("empty event log".to_string()))?
        .map_err(|e| SimError::Validation(format!("event log read: {}", e)))?;
    if magic != TSV_MAGIC {
        return Err(SimError::Validation(format!(
            "event log: bad magic line '{}', expected '{}'",
            magic, TSV_MAGIC
        )));
    }
    let mut initial_pools = None;
    let mut transitions = None;
    for _ in 0..2 {
        let line = lines
            .next()
            .ok_or_else(|| SimError::Validation("event log: truncated metadata".to_string()))?
            .map_err(|e| SimError::Validation(format!("event log read: {}", e)))?;
        parse_tsv_meta_line(&line, &mut initial_pools, &mut transitions)?;
    }
    Ok((
        initial_pools
            .ok_or_else(|| SimError::Validation("event log: missing initial_pools".to_string()))?,
        transitions
            .ok_or_else(|| SimError::Validation("event log: missing transitions".to_string()))?,
    ))
}

/// Stream a TSV event log's rows, calling `f` per [`EventRecord`] in file order.
/// Skips the 4 header lines (magic, two metadata, column header); the metadata
/// is read separately by [`read_metadata_tsv`]. Bounded memory: one line at a
/// time.
fn for_each_event_tsv(
    path: &Path,
    mut f: impl FnMut(EventRecord) -> Result<(), SimError>,
) -> Result<(), SimError> {
    let file = File::open(path)
        .map_err(|e| SimError::Validation(format!("read event log {}: {}", path.display(), e)))?;
    let mut lineno = 0usize;
    let mut data_started = false;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| SimError::Validation(format!("event log read: {}", e)))?;
        lineno += 1;
        if !data_started {
            // 4 header lines: magic, # initial_pools, # transitions, columns.
            if lineno >= 4 {
                data_started = true;
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        f(parse_event_fields(&fields, lineno)?)?;
    }
    Ok(())
}

/// Write the event log in the requested format.
pub fn write(log: &EventLog, path: &Path, format: LineListFormat) -> Result<(), SimError> {
    match format {
        LineListFormat::Tsv => write_tsv(log, path),
        LineListFormat::Parquet => {
            #[cfg(feature = "lineage-parquet")]
            {
                parquet_impl::write_parquet(log, path)
            }
            #[cfg(not(feature = "lineage-parquet"))]
            {
                let _ = (log, path);
                Err(SimError::Validation(
                    "Parquet event-log output requires the 'lineage-parquet' cargo feature; \
                     use --tsv for the dependency-free format."
                        .to_string(),
                ))
            }
        }
    }
}

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Read an event log's metadata (t=0 pools + route table) without its rows,
/// auto-detecting format by extension. Cheap: reads the Parquet footer / the
/// TSV header only.
pub fn read_metadata(path: &Path) -> Result<EventLogMeta, SimError> {
    match ext_of(path).as_str() {
        "tsv" => read_metadata_tsv(path),
        "parquet" => {
            #[cfg(feature = "lineage-parquet")]
            {
                parquet_impl::read_metadata_parquet(path)
            }
            #[cfg(not(feature = "lineage-parquet"))]
            {
                Err(SimError::Validation(
                    "reading Parquet event logs requires the 'lineage-parquet' cargo feature."
                        .to_string(),
                ))
            }
        }
        other => Err(SimError::Validation(format!(
            "cannot infer event-log format from extension '.{}'; use a .tsv or .parquet file",
            other
        ))),
    }
}

/// Stream an event log's rows, calling `f` per [`EventRecord`] in recorded
/// (file) order, auto-detecting format by extension. Bounded memory — the whole
/// log is never materialised (Parquet row group by row group, TSV line by
/// line). Pair with [`read_metadata`] to recover the route table first.
pub fn for_each_event(
    path: &Path,
    f: impl FnMut(EventRecord) -> Result<(), SimError>,
) -> Result<(), SimError> {
    match ext_of(path).as_str() {
        "tsv" => for_each_event_tsv(path, f),
        "parquet" => {
            #[cfg(feature = "lineage-parquet")]
            {
                parquet_impl::for_each_event_parquet(path, f)
            }
            #[cfg(not(feature = "lineage-parquet"))]
            {
                let _ = f;
                Err(SimError::Validation(
                    "reading Parquet event logs requires the 'lineage-parquet' cargo feature."
                        .to_string(),
                ))
            }
        }
        other => Err(SimError::Validation(format!(
            "cannot infer event-log format from extension '.{}'; use a .tsv or .parquet file",
            other
        ))),
    }
}

#[cfg(feature = "lineage-parquet")]
pub use parquet_impl::{read_parquet, write_parquet};

#[cfg(feature = "lineage-parquet")]
mod parquet_impl {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Array, BooleanArray, Float64Array, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    const KV_INITIAL_POOLS: &str = "camdl.event_log.initial_pools";
    const KV_TRANSITIONS: &str = "camdl.event_log.transitions";

    /// Rows per Arrow record batch / Parquet row group. Mirrors the line-list
    /// writer: a few thousand rows keeps the in-RAM column buffers small (the
    /// writer never materialises columns for the whole log at once) and lets the
    /// reader stream row-group by row-group.
    pub(super) const BATCH_ROWS: usize = 8192;

    fn batch_for(events: &[EventRecord], schema: &Arc<Schema>) -> Result<RecordBatch, SimError> {
        let time: Vec<f64> = events.iter().map(|e| e.time).collect();
        let transition: Vec<u64> = events.iter().map(|e| e.transition.0 as u64).collect();
        let multiplicity: Vec<u64> = events.iter().map(|e| e.multiplicity).collect();
        let batched: Vec<bool> = events.iter().map(|e| e.batched).collect();
        let step: Vec<u64> = events.iter().map(|e| e.step).collect();
        let weights: Vec<Option<String>> = events
            .iter()
            .map(|e| e.lineage_weights.as_ref().map(|v| serde_json::to_string(v).unwrap()))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(time)),
                Arc::new(UInt64Array::from(transition)),
                Arc::new(UInt64Array::from(multiplicity)),
                Arc::new(BooleanArray::from(batched)),
                Arc::new(UInt64Array::from(step)),
                Arc::new(StringArray::from(weights)),
            ],
        )
        .map_err(|e| SimError::Validation(format!("event log batch build: {}", e)))
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("time", DataType::Float64, false),
            Field::new("transition", DataType::UInt64, false),
            Field::new("multiplicity", DataType::UInt64, false),
            Field::new("batched", DataType::Boolean, false),
            Field::new("step", DataType::UInt64, false),
            // JSON array of f64 at lineage events; null otherwise.
            Field::new("lineage_weights", DataType::Utf8, true),
        ])
    }

    pub fn write_parquet(log: &EventLog, path: &Path) -> Result<(), SimError> {
        let file = File::create(path).map_err(|e| {
            SimError::Validation(format!("cannot create event log {}: {}", path.display(), e))
        })?;
        let schema = Arc::new(schema());
        let pools_json = serde_json::to_string(&log.initial_pools)
            .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
        let routes_json = serde_json::to_string(&log.transitions)
            .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![
                parquet::file::metadata::KeyValue::new(KV_INITIAL_POOLS.into(), pools_json),
                parquet::file::metadata::KeyValue::new(KV_TRANSITIONS.into(), routes_json),
            ]))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| SimError::Validation(format!("event log parquet init: {}", e)))?;

        // One row group per BATCH_ROWS chunk: bounded column buffers, and the
        // file is readable row-group by row-group.
        for chunk in log.events.chunks(BATCH_ROWS) {
            let batch = batch_for(chunk, &schema)?;
            writer
                .write(&batch)
                .map_err(|e| SimError::Validation(format!("event log batch write: {}", e)))?;
        }
        writer
            .close()
            .map_err(|e| SimError::Validation(format!("event log parquet close: {}", e)))?;
        Ok(())
    }

    /// Extract the route table + t=0 pools from the file's key-value metadata
    /// (the Parquet footer).
    fn read_kv_metadata(
        builder: &ParquetRecordBatchReaderBuilder<File>,
    ) -> Result<EventLogMeta, SimError> {
        let kv = builder.metadata().file_metadata().key_value_metadata();
        let find = |key: &str| -> Option<String> {
            kv.and_then(|m| m.iter().find(|x| x.key == key).and_then(|x| x.value.clone()))
        };
        let pools_json = find(KV_INITIAL_POOLS).ok_or_else(|| {
            SimError::Validation("event log parquet: missing initial_pools metadata".to_string())
        })?;
        let routes_json = find(KV_TRANSITIONS).ok_or_else(|| {
            SimError::Validation("event log parquet: missing transitions metadata".to_string())
        })?;
        let initial_pools = serde_json::from_str(&pools_json)
            .map_err(|e| SimError::Validation(format!("event log initial_pools: {}", e)))?;
        let transitions = serde_json::from_str(&routes_json)
            .map_err(|e| SimError::Validation(format!("event log transitions: {}", e)))?;
        Ok((initial_pools, transitions))
    }

    /// Decode one record batch's rows into [`EventRecord`]s, calling `f` per row.
    fn decode_batch(
        batch: &RecordBatch,
        f: &mut impl FnMut(EventRecord) -> Result<(), SimError>,
    ) -> Result<(), SimError> {
        let time = batch.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
        let transition = batch.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
        let multiplicity = batch.column(2).as_any().downcast_ref::<UInt64Array>().unwrap();
        let batched = batch.column(3).as_any().downcast_ref::<BooleanArray>().unwrap();
        let step = batch.column(4).as_any().downcast_ref::<UInt64Array>().unwrap();
        let weights = batch.column(5).as_any().downcast_ref::<StringArray>().unwrap();
        for r in 0..batch.num_rows() {
            let lineage_weights = if weights.is_null(r) {
                None
            } else {
                Some(serde_json::from_str::<Vec<f64>>(weights.value(r)).map_err(|e| {
                    SimError::Validation(format!("event log lineage_weights: {}", e))
                })?)
            };
            f(EventRecord {
                time: time.value(r),
                transition: TransitionId(transition.value(r) as usize),
                multiplicity: multiplicity.value(r),
                batched: batched.value(r),
                step: step.value(r),
                lineage_weights,
            })?;
        }
        Ok(())
    }

    fn open_builder(path: &Path) -> Result<ParquetRecordBatchReaderBuilder<File>, SimError> {
        let file = File::open(path)
            .map_err(|e| SimError::Validation(format!("open event log {}: {}", path.display(), e)))?;
        ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| SimError::Validation(format!("event log parquet reader: {}", e)))
    }

    pub fn read_parquet(path: &Path) -> Result<EventLog, SimError> {
        let builder = open_builder(path)?;
        let (initial_pools, transitions) = read_kv_metadata(&builder)?;
        let reader = builder
            .build()
            .map_err(|e| SimError::Validation(format!("event log parquet build: {}", e)))?;
        let mut events = Vec::new();
        for batch in reader {
            let batch =
                batch.map_err(|e| SimError::Validation(format!("event log batch: {}", e)))?;
            decode_batch(&batch, &mut |rec| {
                events.push(rec);
                Ok(())
            })?;
        }
        Ok(EventLog { initial_pools, transitions, events })
    }

    /// Read only the metadata (footer), without decoding any rows.
    pub fn read_metadata_parquet(path: &Path) -> Result<EventLogMeta, SimError> {
        read_kv_metadata(&open_builder(path)?)
    }

    /// Stream rows row-group by row-group, calling `f` per [`EventRecord`].
    /// Bounded memory: one batch (≤ `BATCH_ROWS`) decoded at a time.
    pub fn for_each_event_parquet(
        path: &Path,
        mut f: impl FnMut(EventRecord) -> Result<(), SimError>,
    ) -> Result<(), SimError> {
        let reader = open_builder(path)?
            .build()
            .map_err(|e| SimError::Validation(format!("event log parquet build: {}", e)))?;
        for batch in reader {
            let batch =
                batch.map_err(|e| SimError::Validation(format!("event log batch: {}", e)))?;
            decode_batch(&batch, &mut f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::event_log::RouteInfo;

    fn sample_log() -> EventLog {
        use crate::lineage::{CompartmentId, DemeId, TransitionId};
        EventLog {
            initial_pools: vec![
                (DemeId(0), CompartmentId(1), 3),
                (DemeId(1), CompartmentId(4), 2),
            ],
            transitions: vec![
                RouteInfo {
                    source: Some(CompartmentId(0)),
                    source_deme: DemeId(0),
                    destination: Some(CompartmentId(1)),
                    destination_deme: DemeId(0),
                    child_deme: DemeId(0),
                    touches_tracked: true,
                    parent_pools: vec![(CompartmentId(1), DemeId(0)), (CompartmentId(4), DemeId(1))],
                },
                RouteInfo {
                    source: Some(CompartmentId(1)),
                    source_deme: DemeId(0),
                    destination: Some(CompartmentId(2)),
                    destination_deme: DemeId(0),
                    child_deme: DemeId(0),
                    touches_tracked: true,
                    parent_pools: vec![],
                },
            ],
            events: vec![
                EventRecord {
                    time: 0.5,
                    transition: TransitionId(0),
                    multiplicity: 1,
                    batched: false,
                    step: 1,
                    lineage_weights: Some(vec![2.5, 0.3]),
                },
                EventRecord {
                    time: 1.2,
                    transition: TransitionId(1),
                    multiplicity: 3,
                    batched: true,
                    step: 2,
                    lineage_weights: None,
                },
            ],
        }
    }

    #[test]
    fn tsv_round_trip() {
        let log = sample_log();
        let path = std::env::temp_dir().join(format!("camdl_evlog_{}.tsv", std::process::id()));
        write_tsv(&log, &path).unwrap();
        let back = read_tsv(&path).unwrap();
        assert_eq!(log, back);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "lineage-parquet")]
    #[test]
    fn parquet_round_trip() {
        let log = sample_log();
        let path =
            std::env::temp_dir().join(format!("camdl_evlog_{}.parquet", std::process::id()));
        write_parquet(&log, &path).unwrap();
        let back = read_parquet(&path).unwrap();
        assert_eq!(log, back);
        std::fs::remove_file(&path).ok();
    }

    /// A log larger than `parquet_impl::BATCH_ROWS` must round-trip across
    /// multiple row groups in original (time) order. Locks in the chunked
    /// writer so a regression to a single giant batch is caught.
    #[cfg(feature = "lineage-parquet")]
    #[test]
    fn parquet_round_trip_multiple_row_groups() {
        let mut log = sample_log();
        let n = parquet_impl::BATCH_ROWS * 2 + 7; // spans >2 row groups
        log.events = (0..n)
            .map(|i| EventRecord {
                time: i as f64 * 0.5,
                transition: crate::lineage::TransitionId(i % 2),
                multiplicity: 1,
                batched: false,
                step: i as u64,
                lineage_weights: if i % 2 == 0 { Some(vec![1.0, 2.0]) } else { None },
            })
            .collect();
        let path = std::env::temp_dir()
            .join(format!("camdl_evlog_multi_{}.parquet", std::process::id()));
        write_parquet(&log, &path).unwrap();
        let back = read_parquet(&path).unwrap();
        assert_eq!(log.events.len(), back.events.len());
        assert_eq!(log, back, "multi-row-group round trip must preserve order and values");
        std::fs::remove_file(&path).ok();
    }
}
