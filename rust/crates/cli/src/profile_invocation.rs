//! One `camdl profile` invocation's record and heartbeat, under
//! `profiles/<base>/invocations/<invocation_id>/`.
//!
//! A profile's leaves appear one start at a time, each when its start
//! finishes, and the base segment holds no identity-bearing record (several
//! invocations — another method, more seeds, a rerun — share one base). So the
//! tree alone cannot say which grid an invocation will run, how many of its
//! starts are done, or whether it is still running. This directory says all
//! three:
//!
//! - `invocation.json` ([`InvocationRecord`]), written once, atomically, before
//!   the first job runs: the grid, `n_starts`, the seeds, every cell's exact
//!   focal values and the `<point>` directory its leaves are written under.
//! - `progress.json`, the heartbeat every fit method writes (`io::progress`,
//!   gh#278), in phase `profiling`: `step` is the jobs finished, cached ones
//!   included, of `total_jobs`.
//!
//! Neither file enters any hash or any record: the directory is outside every
//! leaf, and nothing here is read back by a run.

use std::path::{Path, PathBuf};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

/// The directory under a profile base that holds one subdirectory per
/// invocation.
pub const INVOCATIONS_DIR: &str = "invocations";

/// The invocation record's filename.
pub const INVOCATION_FILE: &str = "invocation.json";

/// The record's `schema` tag. A changed meaning or a removed field is a new
/// version; a new field is not.
pub const INVOCATION_SCHEMA: &str = "camdl.profile-invocation/v1";

/// One focal axis of the grid: its parameter and its values, in `--sweep`
/// order.
#[derive(Debug, Clone, Serialize)]
pub struct FocalAxis {
    pub name: String,
    pub values: Vec<f64>,
}

/// One grid cell's focal values, serialized as a JSON object in focal-axis
/// order (`{"gamma": 0.26, "beta": 0.84}`).
#[derive(Debug, Clone)]
pub struct Coords(pub Vec<(String, f64)>);

impl Serialize for Coords {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (name, v) in &self.0 {
            m.serialize_entry(name, v)?;
        }
        m.end()
    }
}

/// One grid cell of the invocation and the `<point>` directory, directly under
/// the base, that its leaves are written under.
#[derive(Debug, Clone, Serialize)]
pub struct InvocationPoint {
    pub coords: Coords,
    pub dir: String,
}

/// `invocation.json`: what one `camdl profile` invocation will run.
#[derive(Debug, Clone, Serialize)]
pub struct InvocationRecord {
    pub schema: &'static str,
    pub invocation_id: String,
    /// Unix epoch seconds at which the invocation started.
    pub started_at: u64,
    pub label: Option<String>,
    /// `if2`, `nl-sbplx`, `nl-bobyqa` or `nl-lbfgs`.
    pub algorithm: String,
    pub focal: Vec<FocalAxis>,
    pub n_starts: usize,
    pub seeds: Vec<u64>,
    pub points: Vec<InvocationPoint>,
    /// The `<method>` directory, under each point's `dir`, that this
    /// invocation's leaves are written in. Every job of an invocation shares
    /// one method level; another invocation over the same grid with other
    /// method settings writes beside it, so a reader counting a cell's
    /// finished starts counts under `<dir>/<method_dir>/`.
    pub method_dir: String,
    /// `points.len() × n_starts × seeds.len()`: the heartbeat's `total`.
    pub total_jobs: usize,
}

/// A fresh invocation id, unique per invocation and sortable by start time:
/// `<started_at>-<pid>`. Not identity-bearing.
pub fn invocation_id(started_at: u64) -> String {
    format!("{started_at:010}-{}", std::process::id())
}

/// Unix epoch seconds now.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Create `<base_seg>/invocations/<id>/`, write `invocation.json` into it
/// atomically (temp file + rename), and start its `profiling` heartbeat. The
/// id is the record's, suffixed `-2`, `-3`, … should that directory already
/// exist; the record written carries the id actually used.
///
/// # Errors
/// The directory or the record cannot be written.
pub fn begin(
    base_seg: &Path,
    mut record: InvocationRecord,
    interval: std::time::Duration,
) -> std::io::Result<(PathBuf, io::HeartbeatGuard)> {
    let parent = base_seg.join(INVOCATIONS_DIR);
    std::fs::create_dir_all(&parent)?;
    let base_id = record.invocation_id.clone();
    let mut n = 1;
    let dir = loop {
        let id = if n == 1 { base_id.clone() } else { format!("{base_id}-{n}") };
        let dir = parent.join(&id);
        match std::fs::create_dir(&dir) {
            Ok(()) => {
                record.invocation_id = id;
                break dir;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
            Err(e) => return Err(e),
        }
    };
    let mut json = serde_json::to_vec_pretty(&record).map_err(std::io::Error::other)?;
    json.push(b'\n');
    let tmp = dir.join(format!("{INVOCATION_FILE}.tmp"));
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, dir.join(INVOCATION_FILE))?;
    let hb = io::Heartbeat::profiling(dir.clone(), record.total_jobs as u64, interval);
    Ok((dir, io::HeartbeatGuard::new(hb)))
}

/// The invocation heartbeat's step counter: jobs finished, cached ones
/// included. Each finished job is published at once, between the timer's
/// writes, so a reader sees every job land.
pub struct JobProgress {
    hb: std::sync::Arc<io::HeartbeatGuard>,
    finished: std::sync::atomic::AtomicU64,
}

impl JobProgress {
    /// Start counting at `already_finished` — the jobs the store already
    /// holds, which a resumed invocation counts as done — and publish it.
    pub fn new(hb: std::sync::Arc<io::HeartbeatGuard>, already_finished: u64) -> JobProgress {
        hb.bump(already_finished);
        hb.publish();
        JobProgress { hb, finished: std::sync::atomic::AtomicU64::new(already_finished) }
    }

    /// One more job finished, whatever its outcome: stored, refused because
    /// another process holds its leaf, or already stored by the time it
    /// claimed.
    pub fn job_finished(&self) {
        let n = self.finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        self.hb.bump(n);
        self.hb.publish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> InvocationRecord {
        InvocationRecord {
            schema: INVOCATION_SCHEMA,
            invocation_id: invocation_id(1_789_000_000),
            started_at: 1_789_000_000,
            label: None,
            algorithm: "nl-sbplx".into(),
            focal: vec![FocalAxis { name: "gamma".into(), values: vec![0.26, 0.3] }],
            n_starts: 2,
            seeds: vec![1],
            points: vec![InvocationPoint {
                coords: Coords(vec![("gamma".into(), 0.26), ("beta".into(), 0.84)]),
                dir: "gamma_0.2600__beta_0.8400-12d25783".into(),
            }],
            method_dir: "nlopt_sbplx_-82833741".into(),
            total_jobs: 4,
        }
    }

    /// The field names and the coords' axis order are what readers key on.
    #[test]
    fn the_record_serializes_with_its_stated_fields() {
        let v = serde_json::to_value(record()).unwrap();
        assert_eq!(v["schema"], "camdl.profile-invocation/v1");
        assert_eq!(v["label"], serde_json::Value::Null);
        assert_eq!(v["points"][0]["dir"], "gamma_0.2600__beta_0.8400-12d25783");
        let text = serde_json::to_string(&record()).unwrap();
        assert!(
            text.contains(r#""coords":{"gamma":0.26,"beta":0.84}"#),
            "coords keep the focal-axis order: {text}"
        );
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        for k in [
            "schema", "invocation_id", "started_at", "label", "algorithm", "focal",
            "n_starts", "seeds", "points", "method_dir", "total_jobs",
        ] {
            assert!(keys.contains(&k), "missing {k}: {keys:?}");
        }
    }

    /// A second invocation that lands on the same id gets its own directory.
    #[test]
    fn a_colliding_id_gets_a_suffix() {
        let base = tempfile::tempdir().unwrap();
        let interval = std::time::Duration::from_secs(60);
        let (d1, g1) = begin(base.path(), record(), interval).unwrap();
        let (d2, g2) = begin(base.path(), record(), interval).unwrap();
        assert_ne!(d1, d2);
        let r2: serde_json::Value =
            serde_json::from_slice(&std::fs::read(d2.join(INVOCATION_FILE)).unwrap()).unwrap();
        assert_eq!(
            r2["invocation_id"].as_str().unwrap(),
            d2.file_name().unwrap().to_str().unwrap(),
            "the record names the directory it is in"
        );
        g1.done();
        g2.done();
    }

    fn state(dir: &Path) -> serde_json::Value {
        let p: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("progress.json")).unwrap()).unwrap();
        p["state"].clone()
    }

    /// The heartbeat goes running → done, a step is one job, and a resumed
    /// invocation's cached jobs count as finished from its first write. The
    /// timer interval is far longer than the test, so every write seen here is
    /// a `publish`.
    #[test]
    fn the_heartbeat_counts_cached_jobs_then_each_job_then_ends_done() {
        let base = tempfile::tempdir().unwrap();
        let (dir, guard) =
            begin(base.path(), record(), std::time::Duration::from_secs(3600)).unwrap();
        let hb = std::sync::Arc::new(guard);
        let progress = JobProgress::new(std::sync::Arc::clone(&hb), 3);
        assert_eq!(
            state(&dir),
            serde_json::json!({"running": {"phase": "profiling", "step": 3, "total": 4}}),
        );
        progress.job_finished();
        assert_eq!(
            state(&dir),
            serde_json::json!({"running": {"phase": "profiling", "step": 4, "total": 4}}),
        );
        hb.done();
        hb.commit();
        assert_eq!(state(&dir), serde_json::json!("done"));
        // A write after the terminal one changes nothing.
        progress.job_finished();
        assert_eq!(state(&dir), serde_json::json!("done"));
    }
}
