//! `camdl data split` — split a TSV by time threshold or fraction.
//!
//! Given the model and the stream the file holds (`--model`, `--stream`), the
//! file is read by the fit's own row reader
//! ([`crate::fit::observed_rows::observed_rows`]) and a row trains when the end
//! of what it measures is at or before the split time: the rule of
//! `holdout_after` ([`crate::fit::training_partition::trains_under_cutoff`]),
//! so the training file holds the rows a fit with `holdout_after` at the same
//! time trains on (gh#1138). Without a model the split compares the time
//! label, which agrees with that rule only for a stream labelled by the end of
//! what each row measures.

use std::io::Write;
use std::path::Path;

use crate::caltime_load::TemporalCell;
use crate::fit::observed_rows::ObservedRow;

/// The two halves of a split file, and what the summary says about them.
struct Split<'a> {
    /// The lines written at the top of both files: the header, and any
    /// leading comment lines the reader skips before it.
    preamble: Vec<&'a str>,
    train: Vec<&'a str>,
    holdout: Vec<&'a str>,
    /// The first summary line: the split time and what it was compared with.
    headline: String,
    train_range: String,
    holdout_range: String,
    /// Said after the summary when the split may disagree with the fit.
    note: Option<String>,
}

pub fn cmd_data_split(a: &crate::args::DataSplitArgs) {
    let out = crate::out::Out::new(false);
    if a.at_time.is_none() && a.fraction.is_none() {
        out.fail("error: specify --at-time T or --fraction F (e.g., --fraction 0.7)");
    }
    if a.fraction.is_some_and(|f| f <= 0.0 || f >= 1.0) {
        out.fail("error: --fraction must be between 0 and 1 (e.g., 0.7)");
    }

    let stem = a.file.to_str().unwrap_or_default().trim_end_matches(".tsv");
    let train_path = a.train.clone()
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{}_train.tsv", stem)));
    let holdout_path = a.holdout.clone()
        .unwrap_or_else(|| std::path::PathBuf::from(format!("{}_holdout.tsv", stem)));

    let content = std::fs::read_to_string(&a.file)
        .unwrap_or_else(|e| { out.fail(format!("cannot read {}: {}", a.file.display(), e)); });

    let split = match &a.model {
        Some(model) => split_by_measure(&a.file, &content, model, a.stream.as_deref(),
            a.at_time, a.fraction),
        None => split_by_label(&content, a.time_col.as_deref(), a.at_time, a.fraction),
    }.unwrap_or_else(|e| out.fail(format!("error: {e}")));

    for (path, rows) in [(&train_path, &split.train), (&holdout_path, &split.holdout)] {
        let mut f = std::fs::File::create(path)
            .unwrap_or_else(|e| { out.fail(format!("cannot create {}: {}", path.display(), e)); });
        for line in split.preamble.iter().chain(rows.iter()) {
            writeln!(f, "{}", line).unwrap();
        }
    }

    out.say(&split.headline);
    out.say(format_args!("  Train:   {} observations, t ∈ {}", split.train.len(), split.train_range));
    out.say(format_args!("  Holdout: {} observations, t ∈ {}", split.holdout.len(), split.holdout_range));
    out.say(format_args!("  Written: {}, {}", train_path.display(), holdout_path.display()));
    if let Some(note) = &split.note {
        out.say(format_args!("{} {note}", crate::out::paint::note_tag()));
    }
}

/// The number of training rows `--fraction f` keeps out of `n`.
fn fraction_rows(n: usize, f: f64) -> Result<usize, String> {
    let n_train = (n as f64 * f).round() as usize;
    if n_train == 0 || n_train >= n {
        return Err(format!("--fraction {} gives {} train rows out of {} total", f, n_train, n));
    }
    Ok(n_train)
}

/// Split on each row's time label: a row trains when its label is at or before
/// the split time.
fn split_by_label<'a>(
    content: &'a str,
    time_col: Option<&str>,
    at_time: Option<f64>,
    fraction: Option<f64>,
) -> Result<Split<'a>, String> {
    let mut lines = content.lines();
    let header = lines.next().ok_or("empty file")?;

    // Find time column index
    let cols: Vec<&str> = header.split('\t').collect();
    let time_idx = if let Some(name) = time_col {
        cols.iter().position(|&c| c == name).ok_or_else(|| {
            format!("column '{}' not found. Available: {}", name, cols.join(", "))
        })?
    } else {
        // Auto-detect: look for "time" or "t"
        cols.iter().position(|&c| c == "time" || c == "t").ok_or_else(|| {
            format!("no 'time' or 't' column found. Use --time-col to specify.\n  \
                Available columns: {}", cols.join(", "))
        })?
    };

    // Collect all rows with their time values
    let rows = lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split('\t').nth(time_idx)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|time| (line, time))
                .ok_or_else(|| format!(
                    "cannot parse time from column {} of line: {}", time_idx, line))
        })
        .collect::<Result<Vec<(&str, f64)>, String>>()?;

    // Determine the split threshold
    let threshold = match (at_time, fraction) {
        (Some(t), _) => t,
        // Threshold = time of the last training row
        (None, Some(f)) => rows[fraction_rows(rows.len(), f)? - 1].1,
        (None, None) => return Err("specify --at-time T or --fraction F".into()),
    };

    let (train, holdout): (Vec<_>, Vec<_>) =
        rows.iter().partition(|&&(_, time)| time <= threshold);
    let range = |rows: &[(&str, f64)]| match (rows.first(), rows.last()) {
        (Some((_, lo)), Some((_, hi))) => format!("[{}, {}]", lo, hi),
        _ => "—".into(),
    };
    Ok(Split {
        preamble: vec![header],
        headline: format!("Split at t = {} (column '{}')", threshold, cols[time_idx]),
        train_range: range(&train),
        holdout_range: range(&holdout),
        train: train.iter().map(|(l, _)| *l).collect(),
        holdout: holdout.iter().map(|(l, _)| *l).collect(),
        note: Some(
            "split on the time label. A stream labelled by the start of the period \
             each row measures (covers = day(time), starting_on) keeps a row in the \
             training file that the fit's holdout_after holds out; pass --model and \
             --stream to split as holdout_after does.".into()),
    })
}

/// Split as `holdout_after` partitions a stream: read the file through the
/// fit's row reader, then train each row whose measured end is at or before
/// the split time.
fn split_by_measure<'a>(
    file: &Path,
    content: &'a str,
    model_path: &Path,
    stream: Option<&str>,
    at_time: Option<f64>,
    fraction: Option<f64>,
) -> Result<Split<'a>, String> {
    let utf8 = |p: &Path| p.to_str().map(str::to_owned)
        .ok_or_else(|| format!("path is not valid UTF-8: {}", p.display()));
    let (model, _) = crate::util::load_model(&utf8(model_path)?)?;
    let obs = pick_stream(&model, stream)?;
    if crate::pfilter::is_long_form_stream(obs) {
        return Err(format!(
            "stream '{}' is in long form (one row per stratum), which `data split \
             --model` does not split; declare `holdout_after` in the fit config \
             instead, which needs no split files",
            obs.name));
    }
    let siblings: Vec<&ir::observation::ObservationModel> =
        model.observations.iter().filter(|o| o.source == obs.source).collect();
    let opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        // Nothing is integrated here, so no substep or grid check applies.
        dt: 0.0,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let rows = crate::fit::observed_rows::observed_rows(&utf8(file)?, obs, &siblings, &opts)?;

    let (preamble, body) = data_lines(content);
    if body.len() != rows.len() {
        return Err(format!(
            "internal error: {} has {} data lines but the reader produced {} rows",
            file.display(), body.len(), rows.len()));
    }

    let at = match (at_time, fraction) {
        (Some(t), _) => t,
        // The end of the last training row's period.
        (None, Some(f)) => rows[fraction_rows(rows.len(), f)? - 1].measures.end(),
        (None, None) => return Err("specify --at-time T or --fraction F".into()),
    };

    let (train, holdout): (Vec<_>, Vec<_>) = body.iter().zip(&rows)
        .partition(|(_, row)| crate::fit::training_partition::trains_under_cutoff(row, at));
    let range = |rows: &[(&&str, &ObservedRow)]| match (rows.first(), rows.last()) {
        (Some((_, lo)), Some((_, hi))) => format!("[{}, {}]", written(lo), written(hi)),
        _ => "—".into(),
    };
    Ok(Split {
        preamble,
        headline: format!(
            "Split at t = {at} (stream '{}': a row trains when what it measures ends at \
             or before t = {at}, as under holdout_after)", obs.name),
        train_range: range(&train),
        holdout_range: range(&holdout),
        train: train.iter().map(|(l, _)| **l).collect(),
        holdout: holdout.iter().map(|(l, _)| **l).collect(),
        note: None,
    })
}

/// The observation stream `--stream` names, or the model's only one.
fn pick_stream<'m>(
    model: &'m ir::Model,
    stream: Option<&str>,
) -> Result<&'m ir::observation::ObservationModel, String> {
    let names = || model.observations.iter()
        .map(|o| o.name.as_str()).collect::<Vec<_>>().join(", ");
    match (stream, model.observations.as_slice()) {
        (Some(s), all) => all.iter().find(|o| o.name == s).ok_or_else(|| format!(
            "the model has no observation stream '{s}'; its streams: {}", names())),
        (None, [only]) => Ok(only),
        (None, []) => Err("the model declares no observation stream, so it cannot say \
            what the file's rows measure".into()),
        (None, _) => Err(format!(
            "the model declares several observation streams ({}); name the one the file \
             holds with --stream", names())),
    }
}

/// A file's lines as the observation reader reads them: the leading `#`
/// comment and blank lines and the header, then one line per data row. Blank
/// and `#` lines among the rows are skipped, as the reader skips them, so they
/// reach neither output.
fn data_lines(content: &str) -> (Vec<&str>, Vec<&str>) {
    let skipped = |l: &str| l.trim().is_empty() || l.trim_start().starts_with('#');
    let mut lines = content.lines();
    let mut preamble = Vec::new();
    for line in lines.by_ref() {
        preamble.push(line);
        if !skipped(line) {
            break;
        }
    }
    (preamble, lines.filter(|l| !skipped(l)).collect())
}

/// A row's label cell as the file wrote it.
fn written(row: &ObservedRow) -> String {
    match row.temporal.last() {
        Some(TemporalCell::Number(x)) => x.to_string(),
        Some(TemporalCell::Date(d)) => d.clone(),
        None => row.label.to_string(),
    }
}
