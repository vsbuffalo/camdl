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
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::SimError;

use super::event_log::{EventLog, EventRecord};
use super::writer::LineListFormat;

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

/// Write the event log as TSV.
pub fn write_tsv(log: &EventLog, path: &Path) -> Result<(), SimError> {
    let file = File::create(path).map_err(|e| {
        SimError::Validation(format!("cannot create event log {}: {}", path.display(), e))
    })?;
    let mut out = BufWriter::new(file);
    let w = |out: &mut BufWriter<File>, s: &str| {
        writeln!(out, "{}", s).map_err(|e| SimError::Validation(format!("event log write: {}", e)))
    };

    w(&mut out, TSV_MAGIC)?;
    let pools_json = serde_json::to_string(&log.initial_pools)
        .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
    let routes_json = serde_json::to_string(&log.transitions)
        .map_err(|e| SimError::Validation(format!("event log meta: {}", e)))?;
    w(&mut out, &format!("# initial_pools\t{}", pools_json))?;
    w(&mut out, &format!("# transitions\t{}", routes_json))?;
    w(&mut out, &EVENT_COLUMNS.join("\t"))?;
    for e in &log.events {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}",
            e.time,
            e.transition,
            e.multiplicity,
            e.batched,
            e.step,
            weights_cell(&e.lineage_weights),
        )
        .map_err(|er| SimError::Validation(format!("event log write: {}", er)))?;
    }
    out.flush()
        .map_err(|e| SimError::Validation(format!("event log flush: {}", e)))
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
        if let Some(rest) = line.strip_prefix("# initial_pools\t") {
            initial_pools = Some(
                serde_json::from_str(rest)
                    .map_err(|e| SimError::Validation(format!("event log initial_pools: {}", e)))?,
            );
        } else if let Some(rest) = line.strip_prefix("# transitions\t") {
            transitions = Some(
                serde_json::from_str(rest)
                    .map_err(|e| SimError::Validation(format!("event log transitions: {}", e)))?,
            );
        } else {
            return Err(SimError::Validation(format!(
                "event log: unexpected metadata line '{}'",
                line
            )));
        }
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
        if f.len() != EVENT_COLUMNS.len() {
            return Err(SimError::Validation(format!(
                "event log row {}: expected {} columns, got {}",
                lineno + 5,
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
        events.push(EventRecord { time, transition, multiplicity, batched, step, lineage_weights });
    }

    Ok(EventLog {
        initial_pools: initial_pools
            .ok_or_else(|| SimError::Validation("event log: missing initial_pools".to_string()))?,
        transitions: transitions
            .ok_or_else(|| SimError::Validation("event log: missing transitions".to_string()))?,
        events,
    })
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

/// Read an event log, auto-detecting TSV vs Parquet by extension.
pub fn read(path: &Path) -> Result<EventLog, SimError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "tsv" => read_tsv(path),
        "parquet" => {
            #[cfg(feature = "lineage-parquet")]
            {
                parquet_impl::read_parquet(path)
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

        let time: Vec<f64> = log.events.iter().map(|e| e.time).collect();
        let transition: Vec<u64> = log.events.iter().map(|e| e.transition as u64).collect();
        let multiplicity: Vec<u64> = log.events.iter().map(|e| e.multiplicity).collect();
        let batched: Vec<bool> = log.events.iter().map(|e| e.batched).collect();
        let step: Vec<u64> = log.events.iter().map(|e| e.step).collect();
        let weights: Vec<Option<String>> = log
            .events
            .iter()
            .map(|e| e.lineage_weights.as_ref().map(|v| serde_json::to_string(v).unwrap()))
            .collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(time)),
                Arc::new(UInt64Array::from(transition)),
                Arc::new(UInt64Array::from(multiplicity)),
                Arc::new(BooleanArray::from(batched)),
                Arc::new(UInt64Array::from(step)),
                Arc::new(StringArray::from(weights)),
            ],
        )
        .map_err(|e| SimError::Validation(format!("event log batch build: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| SimError::Validation(format!("event log batch write: {}", e)))?;
        writer
            .close()
            .map_err(|e| SimError::Validation(format!("event log parquet close: {}", e)))?;
        Ok(())
    }

    pub fn read_parquet(path: &Path) -> Result<EventLog, SimError> {
        let file = File::open(path)
            .map_err(|e| SimError::Validation(format!("open event log {}: {}", path.display(), e)))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| SimError::Validation(format!("event log parquet reader: {}", e)))?;

        // Pull the route table + initial pools from key-value metadata.
        let kv = builder.metadata().file_metadata().key_value_metadata();
        let find = |key: &str| -> Option<String> {
            kv.and_then(|m| {
                m.iter()
                    .find(|x| x.key == key)
                    .and_then(|x| x.value.clone())
            })
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

        let reader = builder
            .build()
            .map_err(|e| SimError::Validation(format!("event log parquet build: {}", e)))?;
        let mut events = Vec::new();
        for batch in reader {
            let batch =
                batch.map_err(|e| SimError::Validation(format!("event log batch: {}", e)))?;
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
                    Some(
                        serde_json::from_str::<Vec<f64>>(weights.value(r)).map_err(|e| {
                            SimError::Validation(format!("event log lineage_weights: {}", e))
                        })?,
                    )
                };
                events.push(EventRecord {
                    time: time.value(r),
                    transition: transition.value(r) as usize,
                    multiplicity: multiplicity.value(r),
                    batched: batched.value(r),
                    step: step.value(r),
                    lineage_weights,
                });
            }
        }

        Ok(EventLog { initial_pools, transitions, events })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::event_log::RouteInfo;

    fn sample_log() -> EventLog {
        EventLog {
            initial_pools: vec![(0, 1, 3), (1, 4, 2)],
            transitions: vec![
                RouteInfo {
                    source: Some(0),
                    source_deme: 0,
                    destination: Some(1),
                    destination_deme: 0,
                    child_deme: 0,
                    touches_tracked: true,
                    parent_pools: vec![(1, 0), (4, 1)],
                },
                RouteInfo {
                    source: Some(1),
                    source_deme: 0,
                    destination: Some(2),
                    destination_deme: 0,
                    child_deme: 0,
                    touches_tracked: true,
                    parent_pools: vec![],
                },
            ],
            events: vec![
                EventRecord {
                    time: 0.5,
                    transition: 0,
                    multiplicity: 1,
                    batched: false,
                    step: 1,
                    lineage_weights: Some(vec![2.5, 0.3]),
                },
                EventRecord {
                    time: 1.2,
                    transition: 1,
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
}
