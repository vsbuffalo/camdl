//! Test-only support helpers shared across the crate's unit tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A collision-free unique temp-directory path for a unit test.
///
/// The previous `temp_dir().join(format!("..._{pid}_{nanos}"))` scheme raced
/// between parallel test threads that hit the same nanosecond, producing
/// intermittent filesystem failures under full-suite load (gh#153). A
/// process-wide monotonic counter guarantees cross-thread uniqueness; the pid
/// keeps it unique across concurrent test processes. The directory is NOT
/// created — callers `create_dir_all` (and clean up) as before.
pub(crate) fn unique_temp_dir(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("camdl_{}_{}_{}", prefix, std::process::id(), n))
}
