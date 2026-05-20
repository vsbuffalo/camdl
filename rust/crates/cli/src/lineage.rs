//! CLI glue for the three-layer lineage architecture (Layers 1–2).
//!
//! Entry points:
//!   - [`run_simulate_event_log`]: the `camdl simulate --event-log` path
//!     (Layer 1). Runs the chosen backend with the identity-free event
//!     *recorder* attached, writes the recorded [`sim::lineage::EventLog`] to
//!     disk (TSV / Parquet), and emits the count trajectory (unless
//!     suppressed). The simulation draws no identities.
//!   - [`cmd_lineage_realize`]: the offline `camdl lineage realize` path
//!     (Layer 2). Replays an event log at `--identity-seed`, sampling identity
//!     attributions from the recorded weights, and writes a line list with the
//!     §4a attribution log-probability.
//!   - [`cmd_lineage_tree`] / [`cmd_lineage_sojourn`] / [`cmd_lineage_cohort`]:
//!     offline projections over a realized line list (unchanged).

use std::io::Write;
use std::path::{Path, PathBuf};

use std::collections::HashMap;

use sim::lineage::{
    tree::{select_samples, summarize, Flat, SamplingScheme, Stratified, TransmissionForest},
    DemeId, LineListFormat, LineListWriter, TsvLineListWriter,
};

use crate::args::{
    LineageCohortArgs, LineageRealizeArgs, LineageSojournArgs, LineageTreeArgs, SimulateArgs,
};
use crate::util::SimRun;

/// Resolve the requested artifact format from `--format` / `--tsv`.
/// Default is Parquet (production), per the proposal.
fn resolve_format(tsv: bool, format: &Option<String>) -> Result<LineListFormat, String> {
    if tsv {
        return Ok(LineListFormat::Tsv);
    }
    match format {
        None => Ok(LineListFormat::Parquet),
        Some(s) => LineListFormat::parse(s)
            .ok_or_else(|| format!("unknown --format '{}'; expected 'parquet' or 'tsv'", s)),
    }
}

/// Resolve the artifact format honoring the output path: explicit `--tsv` /
/// `--format` wins, else infer from the explicit output path's extension
/// (so `--event-log foo.tsv` writes TSV, mirroring how the read side
/// auto-detects), else Parquet (production default).
fn resolve_format_with_path(
    tsv: bool,
    format: &Option<String>,
    explicit_out: Option<&Path>,
) -> Result<LineListFormat, String> {
    if tsv || format.is_some() {
        return resolve_format(tsv, format);
    }
    if let Some(p) = explicit_out {
        match p.extension().and_then(|e| e.to_str()) {
            Some("tsv") => return Ok(LineListFormat::Tsv),
            Some("parquet") => return Ok(LineListFormat::Parquet),
            _ => {}
        }
    }
    Ok(LineListFormat::Parquet)
}

/// Default output path for a given format and stem, when no explicit output is
/// requested.
fn default_out(stem: &str, format: LineListFormat) -> PathBuf {
    match format {
        LineListFormat::Tsv => PathBuf::from(format!("{stem}.tsv")),
        LineListFormat::Parquet => PathBuf::from(format!("{stem}.parquet")),
    }
}

/// Build the streamed writer for the chosen format. Parquet is behind the
/// `lineage-parquet` cargo feature; without it, Parquet requests fail with a
/// clear message rather than silently falling back to TSV.
fn build_writer(
    format: LineListFormat,
    path: &Path,
) -> Result<Box<dyn LineListWriter>, String> {
    match format {
        LineListFormat::Tsv => {
            let w = TsvLineListWriter::create(path).map_err(|e| format!("{:?}", e))?;
            Ok(Box::new(w))
        }
        LineListFormat::Parquet => {
            #[cfg(feature = "lineage-parquet")]
            {
                let w = sim::lineage::ParquetLineListWriter::create(path)
                    .map_err(|e| format!("{:?}", e))?;
                Ok(Box::new(w))
            }
            #[cfg(not(feature = "lineage-parquet"))]
            {
                let _ = path;
                Err("Parquet line-list output requires the 'lineage-parquet' \
                     cargo feature, which is not enabled in this build. Use \
                     --tsv for the dependency-free format."
                    .to_string())
            }
        }
    }
}

/// `camdl simulate --event-log` — Layer 1: record the identity-free event log.
pub fn run_simulate_event_log(a: &SimulateArgs, run: &SimRun) {
    let explicit_out = a
        .event_log
        .clone()
        .filter(|p| p.as_os_str() != "auto");
    let format = resolve_format_with_path(a.tsv, &a.format, explicit_out.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    let out_path = explicit_out.unwrap_or_else(|| default_out("event_log", format));

    let (traj, model, event_log, exact) =
        crate::util::run_simulation_event_log(run).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

    sim::lineage::event_log_io::write(&event_log, &out_path, format).unwrap_or_else(|e| {
        eprintln!("error: writing event log: {:?}", e);
        std::process::exit(1);
    });

    let n_lineage = event_log
        .events
        .iter()
        .filter(|e| e.lineage_weights.is_some())
        .count();
    eprintln!(
        "event log: wrote {} events ({} lineage) to {} ({}); {}",
        event_log.events.len(),
        n_lineage,
        out_path.display(),
        match format {
            LineListFormat::Tsv => "tsv",
            LineListFormat::Parquet => "parquet",
        },
        if exact {
            "exact (Gillespie)".to_string()
        } else {
            "batched — sub-dt bias is reported by `lineage realize`".to_string()
        },
    );
    eprintln!(
        "  next: camdl lineage realize {} --identity-seed <N> -o line_list.{}",
        out_path.display(),
        match format {
            LineListFormat::Tsv => "tsv",
            LineListFormat::Parquet => "parquet",
        }
    );

    // Count trajectory output (stdout or --output). The trajectory is
    // byte-identical to a run without --event-log at the same seed.
    let output_path = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
    let suppress_traj = a.obs_only.is_some();
    if suppress_traj {
        return;
    }

    let mut out: Box<dyn Write> = match &output_path {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("error: cannot create {}: {}", p, e);
                std::process::exit(1);
            }),
        )),
        None => Box::new(std::io::stdout()),
    };

    let int_names: Vec<&str> = model
        .compartments
        .iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str())
        .collect();
    let real_names: Vec<&str> = model
        .compartments
        .iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Real)
        .map(|c| c.name.as_str())
        .collect();
    let tr_names: Vec<&str> = model.transitions.iter().map(|t| t.name.as_str()).collect();

    writeln!(out, "# {}", crate::version::VERSION).ok();
    write!(out, "t").ok();
    for n in &int_names {
        write!(out, "\t{}", n).ok();
    }
    for n in &real_names {
        write!(out, "\t{}", n).ok();
    }
    for n in &tr_names {
        write!(out, "\tflow_{}", n).ok();
    }
    writeln!(out).ok();

    for snap in &traj.snapshots {
        write!(out, "{}", snap.t).ok();
        for &c in &snap.int_state.counts {
            write!(out, "\t{}", c).ok();
        }
        for &v in &snap.real_state.values {
            write!(out, "\t{:.4}", v).ok();
        }
        for &f in &snap.flows.counts {
            write!(out, "\t{}", f).ok();
        }
        writeln!(out).ok();
    }
    out.flush().ok();
}

/// `camdl lineage realize EVENT_LOG --identity-seed N -o LINE_LIST` — Layer 2:
/// replay the event log into a line list, drawing identity attributions from
/// the recorded per-pool weights. Different `--identity-seed`s give i.i.d.
/// draws from `P(identities | event log)`.
pub fn cmd_lineage_realize(a: &LineageRealizeArgs) {
    let event_log = sim::lineage::event_log_io::read(&a.event_log).unwrap_or_else(|e| {
        eprintln!("error: reading event log {}: {:?}", a.event_log.display(), e);
        std::process::exit(1);
    });

    let format = resolve_format_with_path(a.tsv, &a.format, a.output.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    let out_path = a
        .output
        .clone()
        .unwrap_or_else(|| default_out("line_list", format));

    let mut writer = build_writer(format, &out_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let summary =
        sim::lineage::realize(&event_log, a.identity_seed, writer.as_mut()).unwrap_or_else(|e| {
            eprintln!("error: realize: {:?}", e);
            std::process::exit(1);
        });

    eprintln!(
        "lineage realize: wrote line list to {} ({}); identity-seed {}, {} edges, \
         log P(line list | event log) = {:.6}",
        out_path.display(),
        match format {
            LineListFormat::Tsv => "tsv",
            LineListFormat::Parquet => "parquet",
        },
        a.identity_seed,
        summary.edges,
        summary.total_logprob,
    );
    if summary.exact {
        eprintln!("  sub-dt bias: 0.000 (exact — Gillespie event log)");
    } else {
        eprintln!(
            "  sub-dt bias: {:.3} (batched event log; shrink --dt or use \
             --backend gillespie at record time for trustworthy trees)",
            summary.sub_dt_fraction
        );
    }
}

/// Validate a sampling rate in `[0, 1]`, naming `what` in the error.
fn parse_rate(s: &str, what: &str) -> Result<f64, String> {
    let rate: f64 = s
        .parse()
        .map_err(|e| format!("invalid {} '{}': {}", what, s, e))?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(format!("{} must be in [0, 1], got {}", what, rate));
    }
    Ok(rate)
}

/// Parse the `stratified:` spec `idx=rate,...,default=rate` keyed on integer
/// deme index. `default` is optional (absent → 0.0 for unlisted demes).
/// Returns `(per-deme rates, default)`.
fn parse_stratified_spec(spec: &str) -> Result<(HashMap<DemeId, f64>, f64), String> {
    let mut rates: HashMap<DemeId, f64> = HashMap::new();
    let mut default = 0.0_f64;
    let mut default_set = false;
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (key, val) = part.split_once('=').ok_or_else(|| {
            format!(
                "stratified entry '{}' must be 'idx=rate' (or 'default=rate')",
                part
            )
        })?;
        let key = key.trim();
        if key.eq_ignore_ascii_case("default") {
            default = parse_rate(val.trim(), "stratified default rate")?;
            default_set = true;
        } else {
            let idx: DemeId = key
                .parse()
                .map_err(|e| format!("invalid deme index '{}': {}", key, e))?;
            let rate = parse_rate(val.trim(), &format!("stratified rate for deme {}", idx))?;
            rates.insert(idx, rate);
        }
    }
    if rates.is_empty() && !default_set {
        return Err(format!(
            "stratified spec '{}' has no rates; expected \
             'idx=rate,...,default=rate' (e.g. stratified:0=0.5,1=0.05,default=0.1)",
            spec
        ));
    }
    Ok((rates, default))
}

/// Parse a `--scheme` string into a sampling scheme. `sim_end` is the
/// simulation horizon (the sampling time for never-removed individuals).
/// Supports `flat:RATE` and `stratified:idx=rate,...,default=rate`.
fn parse_scheme(s: &str, sim_end: f64) -> Result<Box<dyn SamplingScheme>, String> {
    if let Some(rest) = s.strip_prefix("flat:") {
        let rate = parse_rate(rest, "flat sampling rate")?;
        Ok(Box::new(Flat::new(rate, sim_end)))
    } else if let Some(spec) = s.strip_prefix("stratified:") {
        let (rates, default) = parse_stratified_spec(spec)?;
        Ok(Box::new(Stratified::new(rates, default, sim_end)))
    } else {
        Err(format!(
            "unknown sampling scheme '{}'. Supported: 'flat:RATE' \
             (e.g. flat:0.1) and 'stratified:idx=rate,...,default=rate' \
             (e.g. stratified:0=0.5,1=0.05,default=0.1).",
            s
        ))
    }
}

/// Read a line list from a path, auto-detecting TSV vs Parquet by extension.
fn read_line_list(path: &Path) -> Result<Vec<sim::lineage::LineListEntry>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "tsv" => sim::lineage::tree::read_tsv(path).map_err(|e| format!("{:?}", e)),
        "parquet" => {
            #[cfg(feature = "lineage-parquet")]
            {
                sim::lineage::tree::read_parquet(path).map_err(|e| format!("{:?}", e))
            }
            #[cfg(not(feature = "lineage-parquet"))]
            {
                Err("reading Parquet line lists requires the 'lineage-parquet' \
                     cargo feature."
                    .to_string())
            }
        }
        other => Err(format!(
            "cannot infer line-list format from extension '.{}'; use a .tsv or \
             .parquet file",
            other
        )),
    }
}

/// `camdl lineage tree LINE_LIST` — offline transmission-tree projection.
///
/// Sampling draws from **all** individuals (not just chain endpoints); a
/// sampled individual's pendant tip is placed at its removal time (or the
/// simulation horizon if never removed).
pub fn cmd_lineage_tree(a: &LineageTreeArgs) {
    let entries = read_line_list(&a.line_list).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    // Per-individual summaries (deme + removal time) and the horizon — the
    // candidate set is every individual.
    let (summaries, sim_end) = summarize(&entries);

    let scheme = parse_scheme(&a.scheme, sim_end).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let forest = TransmissionForest::from_entries(&entries);
    let mut rng = sim::rng::StatefulRng::new(a.sample_seed);
    let sampled = select_samples(scheme.as_ref(), &summaries, &mut rng);

    if sampled.is_empty() {
        eprintln!(
            "warning: sampling scheme selected 0 tips from {} candidate \
             individuals; no tree to emit",
            summaries.len()
        );
    }

    let trees = forest.prune_to(&sampled);

    // Emit one Newick line per pruned tree (forest → multiple roots possible).
    let mut out: Box<dyn Write> = match &a.output {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("error: cannot create {}: {}", p.display(), e);
                std::process::exit(1);
            }),
        )),
        None => Box::new(std::io::stdout()),
    };
    for t in &trees {
        writeln!(out, "{}", t.to_newick()).ok();
    }
    out.flush().ok();

    eprintln!(
        "lineage tree: {} candidate individuals, {} sampled tips, {} tree(s)",
        summaries.len(),
        sampled.len(),
        trees.len()
    );
}

/// `camdl lineage sojourn LINE_LIST --compartment ID` — dwell-time distribution.
pub fn cmd_lineage_sojourn(a: &LineageSojournArgs) {
    let entries = read_line_list(&a.line_list).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let result = sim::lineage::project::sojourn(&entries, a.compartment);

    // Per-individual sojourns to stdout / --output (TSV).
    let mut out: Box<dyn Write> = match &a.output {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("error: cannot create {}: {}", p.display(), e);
                std::process::exit(1);
            }),
        )),
        None => Box::new(std::io::stdout()),
    };
    writeln!(out, "individual\tentry\texit\tdwell").ok();
    for s in &result.completed {
        writeln!(out, "{}\t{}\t{}\t{}", s.individual, s.entry, s.exit, s.dwell).ok();
    }
    out.flush().ok();

    // Summary to stderr (always).
    eprintln!(
        "lineage sojourn (compartment {}): {} completed, {} right-censored; \
         mean dwell {:.4}, median {:.4}, p90 {:.4}",
        a.compartment,
        result.completed.len(),
        result.censored,
        result.mean_dwell(),
        result.dwell_quantile(0.5),
        result.dwell_quantile(0.9),
    );
}

/// `camdl lineage cohort LINE_LIST --event infection` — per-time-window summary.
pub fn cmd_lineage_cohort(a: &LineageCohortArgs) {
    use sim::lineage::project::CohortEvent;

    let entries = read_line_list(&a.line_list).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    if a.window <= 0.0 {
        eprintln!("error: --window must be positive, got {}", a.window);
        std::process::exit(1);
    }

    // `infection` (the model-independent lineage-event filter) or a transition id.
    let event = if a.event.eq_ignore_ascii_case("infection") {
        CohortEvent::Infection
    } else {
        match a.event.parse::<usize>() {
            Ok(t) => CohortEvent::Transition(t),
            Err(_) => {
                eprintln!(
                    "error: --event must be 'infection' or a transition id (integer), got '{}'",
                    a.event
                );
                std::process::exit(1);
            }
        }
    };

    let bins = sim::lineage::project::cohort(&entries, event, a.window, a.align_zero);

    let mut out: Box<dyn Write> = match &a.output {
        Some(p) => Box::new(std::io::BufWriter::new(
            std::fs::File::create(p).unwrap_or_else(|e| {
                eprintln!("error: cannot create {}: {}", p.display(), e);
                std::process::exit(1);
            }),
        )),
        None => Box::new(std::io::stdout()),
    };
    writeln!(out, "window_start\twindow_end\tincidence\tcumulative").ok();
    for b in &bins {
        writeln!(out, "{}\t{}\t{}\t{}", b.start, b.end, b.incidence, b.cumulative).ok();
    }
    out.flush().ok();

    let total: u64 = bins.last().map_or(0, |b| b.cumulative);
    eprintln!(
        "lineage cohort: {} window(s), {} total events",
        bins.len(),
        total
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_scheme_parses() {
        assert!(parse_scheme("flat:0.1", 10.0).is_ok());
        assert!(parse_scheme("flat:1.0", 10.0).is_ok());
        // Out-of-range and malformed rates are rejected.
        assert!(parse_scheme("flat:1.5", 10.0).is_err());
        assert!(parse_scheme("flat:abc", 10.0).is_err());
    }

    #[test]
    fn stratified_spec_parses_indices_and_default() {
        let (rates, default) =
            parse_stratified_spec("0=0.5,1=0.05,default=0.1").unwrap();
        assert_eq!(rates.get(&0), Some(&0.5));
        assert_eq!(rates.get(&1), Some(&0.05));
        assert_eq!(default, 0.1);
        // Whitespace tolerated.
        let (rates, default) = parse_stratified_spec(" 2 = 0.2 , default = 0.3 ").unwrap();
        assert_eq!(rates.get(&2), Some(&0.2));
        assert_eq!(default, 0.3);
    }

    #[test]
    fn stratified_default_optional_defaults_to_zero() {
        let (rates, default) = parse_stratified_spec("0=0.4").unwrap();
        assert_eq!(rates.get(&0), Some(&0.4));
        assert_eq!(default, 0.0);
    }

    #[test]
    fn stratified_spec_rejects_garbage() {
        assert!(parse_stratified_spec("").is_err()); // no rates
        assert!(parse_stratified_spec("0").is_err()); // missing '='
        assert!(parse_stratified_spec("0=2.0").is_err()); // rate out of range
        assert!(parse_stratified_spec("x=0.1").is_err()); // non-integer index
    }

    #[test]
    fn unknown_scheme_is_rejected() {
        assert!(parse_scheme("random:0.1", 10.0).is_err());
    }
}
