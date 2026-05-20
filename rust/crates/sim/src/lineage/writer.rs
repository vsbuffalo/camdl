//! Streamed, append-only line-list writers.
//!
//! One [`LineListEntry`] is written per identity-tracked event. Records are
//! streamed to disk and never held whole in RAM (the proposal's online
//! constraint). Two formats:
//!
//! - **TSV** ([`TsvLineListWriter`]) — dependency-free, the always-available
//!   baseline. The `--tsv` / `--format tsv` path.
//! - **Parquet** ([`ParquetLineListWriter`]) — production columnar format via
//!   the `parquet` + `arrow` crates, behind the default-on `lineage-parquet`
//!   cargo feature. Rows are buffered into row-group batches and flushed.
//!
//! The offline pruner ([`super::tree`]) reads both formats back.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::SimError;

use super::{CompartmentId, DemeId, IndividualId, ParentRef, TransitionId};

/// One record per identity-tracked event.
#[derive(Debug, Clone, PartialEq)]
pub struct LineListEntry {
    pub time: f64,
    pub transition: TransitionId,
    pub individual: IndividualId,
    pub source: Option<CompartmentId>,
    pub destination: Option<CompartmentId>,
    /// The focal (child) individual's deme — its stratum / patch. For a
    /// lineage event this is the destination stratum `a`; for a simple
    /// transition the source/destination stratum.
    pub deme: DemeId,
    pub parent: ParentRef,
    /// The parent individual's deme — its stratum / patch `b`. Populated only
    /// at lineage events (`parent = Individual(..)`); `None` otherwise. The
    /// pair (`parent_deme`, `deme`) is the contact-structured edge `b → a`.
    pub parent_deme: Option<DemeId>,
}

/// Which on-disk format the line list is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineListFormat {
    Tsv,
    Parquet,
}

impl LineListFormat {
    /// Parse a CLI `--format` value.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "tsv" => Some(LineListFormat::Tsv),
            "parquet" => Some(LineListFormat::Parquet),
            _ => None,
        }
    }
}

/// Streamed, append-only line-list sink. `init` writes the header/schema,
/// `write` appends one record, `finish` flushes and closes.
pub trait LineListWriter {
    fn init(&mut self) -> Result<(), SimError>;
    fn write(&mut self, entry: &LineListEntry) -> Result<(), SimError>;
    fn finish(&mut self) -> Result<(), SimError>;
}

/// Lets the CLI pick the format at runtime (`Box<dyn LineListWriter>`) and
/// still satisfy the generic `LineageObserver<W: LineListWriter>` bound.
impl LineListWriter for Box<dyn LineListWriter> {
    fn init(&mut self) -> Result<(), SimError> {
        (**self).init()
    }
    fn write(&mut self, entry: &LineListEntry) -> Result<(), SimError> {
        (**self).write(entry)
    }
    fn finish(&mut self) -> Result<(), SimError> {
        (**self).finish()
    }
}

/// Encode a `ParentRef` to the two on-disk columns `(parent_kind, parent_id)`.
///
/// `parent_kind` is a stable string tag; `parent_id` is the individual id for
/// the `individual` kind and `-1` (sentinel) otherwise. Keeping the kind
/// explicit means the offline reader never has to infer "is this -1 a real id
/// or a sentinel."
fn parent_columns(p: ParentRef) -> (&'static str, i64) {
    match p {
        ParentRef::Individual(IndividualId(id)) => ("individual", id as i64),
        ParentRef::Import => ("import", -1),
        ParentRef::Seed => ("seed", -1),
        ParentRef::None => ("none", -1),
    }
}

/// Column value for an `Option<CompartmentId>`: the id or `-1` if absent.
fn comp_column(c: Option<CompartmentId>) -> i64 {
    c.map_or(-1, |g| g as i64)
}

/// Column value for an `Option<DemeId>`: the deme or `-1` if absent (a
/// non-lineage event has no parent deme).
fn deme_column(d: Option<DemeId>) -> i64 {
    d.map_or(-1, |g| g as i64)
}

/// The fixed column order shared by both formats (and the tree reader).
pub const COLUMNS: &[&str] = &[
    "time",
    "transition",
    "individual",
    "source",
    "destination",
    "deme",
    "parent_kind",
    "parent_id",
    "parent_deme",
];

// ── TSV ─────────────────────────────────────────────────────────────────────

pub struct TsvLineListWriter {
    out: BufWriter<File>,
}

impl TsvLineListWriter {
    pub fn create(path: &Path) -> Result<Self, SimError> {
        let file = File::create(path).map_err(|e| {
            SimError::Validation(format!("cannot create line list {}: {}", path.display(), e))
        })?;
        Ok(TsvLineListWriter { out: BufWriter::new(file) })
    }
}

impl LineListWriter for TsvLineListWriter {
    fn init(&mut self) -> Result<(), SimError> {
        writeln!(self.out, "{}", COLUMNS.join("\t"))
            .map_err(|e| SimError::Validation(format!("line list write: {}", e)))
    }

    fn write(&mut self, e: &LineListEntry) -> Result<(), SimError> {
        let (kind, pid) = parent_columns(e.parent);
        writeln!(
            self.out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            e.time,
            e.transition,
            e.individual.0,
            comp_column(e.source),
            comp_column(e.destination),
            e.deme,
            kind,
            pid,
            deme_column(e.parent_deme),
        )
        .map_err(|e| SimError::Validation(format!("line list write: {}", e)))
    }

    fn finish(&mut self) -> Result<(), SimError> {
        self.out
            .flush()
            .map_err(|e| SimError::Validation(format!("line list flush: {}", e)))
    }
}

// ── Parquet ───────────────────────────────────────────────────────────────────

#[cfg(feature = "lineage-parquet")]
mod parquet_impl {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray, UInt32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    /// Rows per Arrow record batch / Parquet row group. A few thousand rows
    /// keeps the in-RAM buffer small while amortising encode overhead — the
    /// streaming invariant holds (we never buffer the whole log).
    const BATCH_ROWS: usize = 8192;

    struct Buffer {
        time: Vec<f64>,
        transition: Vec<u64>,
        individual: Vec<u64>,
        source: Vec<i64>,
        destination: Vec<i64>,
        deme: Vec<u32>,
        parent_kind: Vec<&'static str>,
        parent_id: Vec<i64>,
        parent_deme: Vec<i64>,
    }

    impl Buffer {
        fn new() -> Self {
            Buffer {
                time: Vec::with_capacity(BATCH_ROWS),
                transition: Vec::with_capacity(BATCH_ROWS),
                individual: Vec::with_capacity(BATCH_ROWS),
                source: Vec::with_capacity(BATCH_ROWS),
                destination: Vec::with_capacity(BATCH_ROWS),
                deme: Vec::with_capacity(BATCH_ROWS),
                parent_kind: Vec::with_capacity(BATCH_ROWS),
                parent_id: Vec::with_capacity(BATCH_ROWS),
                parent_deme: Vec::with_capacity(BATCH_ROWS),
            }
        }

        fn len(&self) -> usize {
            self.time.len()
        }

        fn clear(&mut self) {
            self.time.clear();
            self.transition.clear();
            self.individual.clear();
            self.source.clear();
            self.destination.clear();
            self.deme.clear();
            self.parent_kind.clear();
            self.parent_id.clear();
            self.parent_deme.clear();
        }
    }

    pub struct ParquetLineListWriter {
        writer: Option<ArrowWriter<File>>,
        schema: Arc<Schema>,
        buf: Buffer,
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("time", DataType::Float64, false),
            Field::new("transition", DataType::UInt64, false),
            Field::new("individual", DataType::UInt64, false),
            Field::new("source", DataType::Int64, false),
            Field::new("destination", DataType::Int64, false),
            Field::new("deme", DataType::UInt32, false),
            Field::new("parent_kind", DataType::Utf8, false),
            Field::new("parent_id", DataType::Int64, false),
            Field::new("parent_deme", DataType::Int64, false),
        ])
    }

    impl ParquetLineListWriter {
        pub fn create(path: &Path) -> Result<Self, SimError> {
            let file = File::create(path).map_err(|e| {
                SimError::Validation(format!(
                    "cannot create line list {}: {}",
                    path.display(),
                    e
                ))
            })?;
            let schema = Arc::new(schema());
            let props = WriterProperties::builder().build();
            let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
                .map_err(|e| SimError::Validation(format!("parquet writer init: {}", e)))?;
            Ok(ParquetLineListWriter {
                writer: Some(writer),
                schema,
                buf: Buffer::new(),
            })
        }

        fn flush_batch(&mut self) -> Result<(), SimError> {
            if self.buf.len() == 0 {
                return Ok(());
            }
            let batch = RecordBatch::try_new(
                self.schema.clone(),
                vec![
                    Arc::new(Float64Array::from(self.buf.time.clone())),
                    Arc::new(UInt64Array::from(self.buf.transition.clone())),
                    Arc::new(UInt64Array::from(self.buf.individual.clone())),
                    Arc::new(Int64Array::from(self.buf.source.clone())),
                    Arc::new(Int64Array::from(self.buf.destination.clone())),
                    Arc::new(UInt32Array::from(self.buf.deme.clone())),
                    Arc::new(StringArray::from(self.buf.parent_kind.clone())),
                    Arc::new(Int64Array::from(self.buf.parent_id.clone())),
                    Arc::new(Int64Array::from(self.buf.parent_deme.clone())),
                ],
            )
            .map_err(|e| SimError::Validation(format!("parquet batch build: {}", e)))?;
            self.writer
                .as_mut()
                .expect("writer present until finish")
                .write(&batch)
                .map_err(|e| SimError::Validation(format!("parquet batch write: {}", e)))?;
            self.buf.clear();
            Ok(())
        }
    }

    impl LineListWriter for ParquetLineListWriter {
        fn init(&mut self) -> Result<(), SimError> {
            Ok(()) // schema set at construction
        }

        fn write(&mut self, e: &LineListEntry) -> Result<(), SimError> {
            let (kind, pid) = parent_columns(e.parent);
            self.buf.time.push(e.time);
            self.buf.transition.push(e.transition as u64);
            self.buf.individual.push(e.individual.0);
            self.buf.source.push(comp_column(e.source));
            self.buf.destination.push(comp_column(e.destination));
            self.buf.deme.push(e.deme);
            self.buf.parent_kind.push(kind);
            self.buf.parent_id.push(pid);
            self.buf.parent_deme.push(deme_column(e.parent_deme));
            if self.buf.len() >= BATCH_ROWS {
                self.flush_batch()?;
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<(), SimError> {
            self.flush_batch()?;
            if let Some(w) = self.writer.take() {
                w.close()
                    .map_err(|e| SimError::Validation(format!("parquet close: {}", e)))?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "lineage-parquet")]
pub use parquet_impl::ParquetLineListWriter;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsv_round_trip_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("camdl_lineage_test_{}.tsv", std::process::id()));
        {
            let mut w = TsvLineListWriter::create(&path).unwrap();
            w.init().unwrap();
            w.write(&LineListEntry {
                time: 1.5,
                transition: 0,
                individual: IndividualId(7),
                source: Some(0),
                destination: Some(1),
                deme: 0,
                parent: ParentRef::Individual(IndividualId(3)),
                parent_deme: Some(1),
            })
            .unwrap();
            w.write(&LineListEntry {
                time: 2.0,
                transition: 2,
                individual: IndividualId(8),
                source: Some(2),
                destination: None,
                deme: 0,
                parent: ParentRef::None,
                parent_deme: None,
            })
            .unwrap();
            w.finish().unwrap();
        }
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines[0], COLUMNS.join("\t"));
        // child deme 0, parent deme 1 (cross-stratum edge 1 → 0).
        assert!(lines[1].starts_with("1.5\t0\t7\t0\t1\t0\tindividual\t3\t1"));
        // non-lineage event: parent_deme sentinel -1.
        assert!(lines[2].starts_with("2\t2\t8\t2\t-1\t0\tnone\t-1\t-1"));
        std::fs::remove_file(&path).ok();
    }
}
