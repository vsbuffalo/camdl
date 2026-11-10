//! `camdl data split` — split a TSV by time threshold or fraction.

use std::io::Write;

pub fn cmd_data_split(a: &crate::args::DataSplitArgs) {
    let out = crate::out::Out::new(false);
    if a.at_time.is_none() && a.fraction.is_none() {
        out.fail("error: specify --at-time T or --fraction F (e.g., --fraction 0.7)");
    }

    let stem = a.file.to_str().unwrap_or_default().trim_end_matches(".tsv");
    let train_path = a.train.clone()
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{}_train.tsv", stem)));
    let holdout_path = a.holdout.clone()
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{}_holdout.tsv", stem)));

    let content = std::fs::read_to_string(&a.file)
        .unwrap_or_else(|e| { out.fail(format!("cannot read {}: {}", a.file.display(), e)); });
    let mut lines = content.lines();
    let header = lines.next().unwrap_or_else(|| {
        out.fail("error: empty file");
    });

    // Find time column index
    let cols: Vec<&str> = header.split('\t').collect();
    let time_idx = if let Some(ref name) = a.time_col {
        cols.iter().position(|&c| c == name).unwrap_or_else(|| {
            out.fail(format!("error: column '{}' not found. Available: {}", name, cols.join(", ")));
        })
    } else {
        // Auto-detect: look for "time" or "t"
        cols.iter().position(|&c| c == "time" || c == "t")
            .unwrap_or_else(|| {
                out.say("error: no 'time' or 't' column found. Use --time-col to specify.");
                out.fail(format!("  Available columns: {}", cols.join(", ")));
            })
    };

    // Collect all rows with their time values
    let rows: Vec<(&str, f64)> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let time: f64 = line.split('\t').nth(time_idx)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or_else(|| {
                    out.fail(format!("error: cannot parse time from column {} of line: {}", time_idx, line));
                });
            (line, time)
        })
        .collect();

    // Determine the split threshold
    let threshold = if let Some(t) = a.at_time {
        t
    } else {
        let f = a.fraction.unwrap();
        if f <= 0.0 || f >= 1.0 {
            out.fail("error: --fraction must be between 0 and 1 (e.g., 0.7)");
        }
        let n_train = (rows.len() as f64 * f).round() as usize;
        if n_train == 0 || n_train >= rows.len() {
            out.fail(format!("error: --fraction {} gives {} train rows out of {} total",
                f, n_train, rows.len()));
        }
        // Threshold = time of the last training row
        rows[n_train - 1].1
    };

    let mut train_rows = Vec::new();
    let mut holdout_rows = Vec::new();
    for &(line, time) in &rows {
        if time <= threshold {
            train_rows.push(line);
        } else {
            holdout_rows.push(line);
        }
    }

    {
        let mut f = std::fs::File::create(&train_path)
            .unwrap_or_else(|e| { out.fail(format!("cannot create {}: {}", train_path.display(), e)); });
        writeln!(f, "{}", header).unwrap();
        for row in &train_rows { writeln!(f, "{}", row).unwrap(); }
    }

    {
        let mut f = std::fs::File::create(&holdout_path)
            .unwrap_or_else(|e| { out.fail(format!("cannot create {}: {}", holdout_path.display(), e)); });
        writeln!(f, "{}", header).unwrap();
        for row in &holdout_rows { writeln!(f, "{}", row).unwrap(); }
    }

    let fmt_range = |rows: &[&str]| -> String {
        let first = rows.first().and_then(|r| r.split('\t').nth(time_idx)?.parse::<f64>().ok());
        let last = rows.last().and_then(|r| r.split('\t').nth(time_idx)?.parse::<f64>().ok());
        match (first, last) {
            (Some(lo), Some(hi)) => format!("[{}, {}]", lo, hi),
            _ => "—".into(),
        }
    };

    out.say(format_args!("Split at t = {} (column '{}')", threshold, cols[time_idx]));
    out.say(format_args!("  Train:   {} observations, t ∈ {}", train_rows.len(), fmt_range(&train_rows)));
    out.say(format_args!("  Holdout: {} observations, t ∈ {}", holdout_rows.len(), fmt_range(&holdout_rows)));
    out.say(format_args!("  Written: {}, {}", train_path.display(), holdout_path.display()));
}
