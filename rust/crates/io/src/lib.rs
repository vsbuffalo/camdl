//! Researcher-facing trajectory I/O.
//!
//! One shared writer for posterior latent-trajectory draws across the
//! particle-based inference methods, emitting the tidy/long TSV format
//! described in `docs/dev/proposals/2026-06-09-latent-trajectory-output-
//! consolidation.md` (§4b). A posterior draw of the latent path *is* a
//! [`sim::Trajectory`] — the same type `simulate` produces — so inference
//! output and `simulate` output share one format, one writer, and one
//! downstream toolchain.

pub mod calendar;
pub mod progress;
pub mod trajectories;

pub use calendar::CalendarMeta;
pub use progress::{
    Heartbeat, Phase, Progress, RunLiveness, RunState, liveness, read_progress, write_progress,
};
pub use trajectories::{
    Granularity, PosteriorDraw, TrajManifest, write_trajectories_tsv,
};
