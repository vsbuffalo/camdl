//! Streaming-realize equivalence (B1): replaying an event log incrementally
//! from disk (`realize_from_path`) must produce a byte-identical line list to
//! the in-memory `realize(&EventLog)`, including across Parquet row-group
//! boundaries (the log here is larger than the writer's `BATCH_ROWS = 8192`,
//! so it spans multiple row groups).

use std::path::{Path, PathBuf};

use sim::lineage::{
    event_log::{EventLog, EventRecord, RouteInfo},
    event_log_io, realize, realize_from_path,
    writer::TsvLineListWriter,
    CompartmentId, DemeId, TransitionId,
};

fn tmp(tag: &str) -> PathBuf {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("camdl_lin_stream_{}_{}_{}", tag, std::process::id(), ns))
}

/// A large synthetic SIR-shaped event log: a `#[lineage]` infection (S=0 → I=1,
/// infector pool I) and a recovery (I=1 → R=2), alternating for many events so
/// the Parquet body spans several row groups.
fn big_log(n_events: usize) -> EventLog {
    let routes = vec![
        RouteInfo {
            source: Some(CompartmentId(0)),
            source_deme: DemeId(0),
            destination: Some(CompartmentId(1)),
            destination_deme: DemeId(0),
            child_deme: DemeId(0),
            touches_tracked: true,
            parent_pools: vec![(CompartmentId(1), DemeId(0))],
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
    ];
    // Seed the I pool so transmissions have parents to sample.
    let initial_pools = vec![(DemeId(0), CompartmentId(1), 200)];
    let events = (0..n_events)
        .map(|i| {
            // 2 transmissions : 1 recovery, so the I pool stays populated.
            if i % 3 == 2 {
                EventRecord {
                    time: i as f64 * 0.1,
                    transition: TransitionId(1),
                    multiplicity: 1,
                    batched: false,
                    step: i as u64,
                    lineage_weights: None,
                }
            } else {
                EventRecord {
                    time: i as f64 * 0.1,
                    transition: TransitionId(0),
                    multiplicity: 1,
                    batched: false,
                    step: i as u64,
                    lineage_weights: Some(vec![1.0]),
                }
            }
        })
        .collect();
    EventLog { initial_pools, transitions: routes, events }
}

fn realize_in_memory(log: &EventLog, seed: u64, out: &Path) {
    let mut w = TsvLineListWriter::create(out).unwrap();
    realize(log, seed, &mut w).unwrap();
}

fn realize_streamed(path: &Path, seed: u64, out: &Path) {
    let mut w = TsvLineListWriter::create(out).unwrap();
    realize_from_path(path, seed, &mut w).unwrap();
}

#[test]
fn streaming_realize_matches_in_memory_across_row_groups() {
    let dir = tmp("eq");
    std::fs::create_dir_all(&dir).unwrap();
    // > 2 row groups at BATCH_ROWS = 8192.
    let log = big_log(20_000);
    let seed = 13;

    // In-memory baseline.
    let ll_mem = dir.join("ll_mem.tsv");
    realize_in_memory(&log, seed, &ll_mem);

    // Stream from a Parquet event log.
    let ev_pq = dir.join("ev.parquet");
    event_log_io::write(&log, &ev_pq, sim::lineage::LineListFormat::Parquet).unwrap();
    let ll_pq = dir.join("ll_pq.tsv");
    realize_streamed(&ev_pq, seed, &ll_pq);

    // Stream from a TSV event log.
    let ev_tsv = dir.join("ev.tsv");
    event_log_io::write(&log, &ev_tsv, sim::lineage::LineListFormat::Tsv).unwrap();
    let ll_tsv = dir.join("ll_tsv.tsv");
    realize_streamed(&ev_tsv, seed, &ll_tsv);

    let mem = std::fs::read(&ll_mem).unwrap();
    let pq = std::fs::read(&ll_pq).unwrap();
    let tsv = std::fs::read(&ll_tsv).unwrap();
    assert!(!mem.is_empty(), "line list should be non-empty");
    assert_eq!(mem, pq, "streaming realize from Parquet must match in-memory realize");
    assert_eq!(mem, tsv, "streaming realize from TSV must match in-memory realize");

    let _ = std::fs::remove_dir_all(&dir);
}
