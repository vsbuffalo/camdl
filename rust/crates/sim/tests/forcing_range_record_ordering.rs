//! An evaluation outside a forcing's range returns only once the evaluation
//! that records it has been reported, on whichever thread it was made. A
//! command that reads its policy's refusal after a computation
//! (`profile`'s points) then finds the refusal of any hold that computation
//! made, even when another thread's evaluation recorded it.
//!
//! This test binary installs its own run-note reporter, which the process
//! keeps, so it holds this one test.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use sim::forcing_range::{DefinedRange, RangeKind};
use sim::run_notes::RunNote;

static REPORTED: AtomicBool = AtomicBool::new(false);
static REPORTS: AtomicUsize = AtomicUsize::new(0);

/// Takes its time before it finishes reporting, so a second thread's
/// evaluation outside the range falls inside the report.
fn slow_reporter(note: &RunNote) {
    if matches!(note, RunNote::ForcingOutsideRange(_)) {
        REPORTS.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(300));
        REPORTED.store(true, Ordering::Release);
    }
}

#[test]
fn an_evaluation_outside_the_range_returns_after_it_is_reported() {
    sim::run_notes::set_reporter(slow_reporter);
    let range = Arc::new(DefinedRange::new("mult", RangeKind::DataTable, 0.0, 50.0, false));
    let start = Arc::new(Barrier::new(2));
    let first = {
        let (range, start) = (range.clone(), start.clone());
        std::thread::spawn(move || {
            start.wait();
            range.note(60.0);
        })
    };
    let second = {
        let (range, start) = (range.clone(), start.clone());
        std::thread::spawn(move || {
            start.wait();
            // Let the first thread begin its report.
            std::thread::sleep(Duration::from_millis(50));
            range.note(70.0);
            REPORTED.load(Ordering::Acquire)
        })
    };
    first.join().unwrap();
    assert!(second.join().unwrap(), "the second evaluation returned before the hold was reported");
    assert_eq!(REPORTS.load(Ordering::Relaxed), 1, "recorded once");
    assert_eq!(range.held().len(), 1, "one side held");
}
