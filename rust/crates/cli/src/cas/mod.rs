//! Content-addressable storage helpers shared between `camdl simulate`
//! (content-addressed by default, one leaf per cell) and `camdl batch run`
//! (bulk experiments). Both write through the same per-cell `CasSink`, so a
//! `simulate` leaf is byte-identical to the `batch` leaf for the same cell.
//!
//! The on-disk store is the `runid` factored layout, rooted at `results/`
//! (the default output root). A leaf's path is one `{label}-{hash8}` segment
//! per identity level in path order; the label is provenance, the `hash8` is
//! identity (see `runid::layout`). For a forward simulation the levels are
//! `model` / `config` / `params` / `scenario` / `seed`
//! (`resolve::resolve_trajectory`):
//!
//! ```text
//! results/
//!   sims/
//!     {model_stem}-{h8}/                      # whole-IR model digest (+ version)
//!       {backend}-dt{dt}-{h8}/                # backend + dt + output schedule
//!         {param_label}-{h8}/                 # resolved base params (+ sweep point)
//!           {scenario}-{h8}/                  # scenario delta (enable/disable/overrides)
//!             seed_{n}-{h8}/
//!               traj.tsv                      # trajectory output (canonical)
//!               run.json                      # RunRecord (kind = sim)
//!               obs/                          # optional, one dir per (obs-model, obs-seed)
//!                 {obs_hash[:8]}-{obs_seed}/
//!                   <stream>.tsv              # observation draws (wide or per-stream)
//!                   obs.json                  # obs metadata (NOT a RunRecord)
//!   fits/
//!     {fit_stem}-{h8}/                        # FitDigest (model + data + fit-wide config)
//!       {NN}-{stage}-{h8}/                    # e.g. 01-scout-{h8}, 02-posterior-{h8}
//!         seed_{n}-{h8}/
//!           run.json                          # RunRecord (kind = fit_stage)
//! ```
//!
//! Browsed uniformly by `camdl list / show / cat`, which read `run.json`
//! (never the path segments).
//!
//! The split between `traj.tsv` (the trajectory leaf) and its
//! `obs/{obs_hash}-{obs_seed}/` child lets users iterate observation draws
//! without recomputing the trajectory: the trajectory identity is the five
//! levels above; the obs child is keyed on `(trajectory run_id, obs_hash,
//! obs_seed)`.

pub mod typed;

// ─── ISO-8601 timestamp helper ───────────────────────────────────────────────

/// Format a SystemTime as ISO 8601 UTC (e.g. "2026-04-16T14:23:11Z").
/// Pure stdlib, no external crate — keeps supply-chain surface zero for
/// this shared module.
pub fn iso8601_utc(t: std::time::SystemTime) -> String {
    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    // Days since 1970-01-01 (epoch day 0).
    let (year, month, day, hour, minute, second) = civil_from_secs(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hour, minute, second)
}

/// Convert a unix timestamp to civil date + time-of-day. The civil-date
/// arithmetic is the canonical proleptic-Gregorian one in `ir::caltime`
/// (`civil_from_unix_epoch_days`); only the HH:MM:SS split is local.
fn civil_from_secs(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let time = secs.rem_euclid(86400) as u32;
    let hour = time / 3600;
    let minute = (time % 3600) / 60;
    let second = time % 60;
    let (year, month, day) = ir::caltime::civil_from_unix_epoch_days(days);
    (year as i32, month as u32, day as u32, hour, minute, second)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso8601_epoch() {
        let epoch = std::time::UNIX_EPOCH;
        assert_eq!(iso8601_utc(epoch), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn iso8601_known_dates() {
        // 2026-04-16T00:00:00Z → 1776297600 unix seconds
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1776297600);
        assert_eq!(iso8601_utc(t), "2026-04-16T00:00:00Z");
        // 2000-01-01T00:00:00Z → 946684800 unix seconds
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(946684800);
        assert_eq!(iso8601_utc(t), "2000-01-01T00:00:00Z");
        // A leap-day: 2024-02-29T12:34:56Z
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1709210096);
        assert_eq!(iso8601_utc(t), "2024-02-29T12:34:56Z");
    }

}
