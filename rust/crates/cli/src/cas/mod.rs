//! Time formatting for the records camdl writes.

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
