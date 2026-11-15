//! The scoring report as a library: the types a model comparison is made of
//! and the pure functions that compute it.
//!
//! This crate takes its inputs as values and returns values. It reads no
//! file, starts no process or thread, reads no clock, environment variable
//! or random number generator, and opens no socket; everything that does
//! stays in `cli`. Its dependency list and a search of its source enforce
//! that (the tests under `tests/`).
//!
//! Design: `docs/dev/proposals/2026-11-14-scoring-report-library.md`.

pub mod combine;
pub mod convergence;
pub mod evidence;
pub mod num;
pub mod preflight;
pub mod trace;
