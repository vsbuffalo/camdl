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

use sim::lineage::{
    tree::{Flat, SamplingScheme, TransmissionForest},
    LineListFormat, LineListWriter, TsvLineListWriter,
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
    let format = resolve_format(a.tsv, &a.format).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let out_path = a
        .event_log
        .clone()
        .filter(|p| p.as_os_str() != "auto")
        .unwrap_or_else(|| default_out("event_log", format));

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

    // Output format: explicit --format / --tsv, else inferred from the output
    // path extension, else Parquet.
    let format = if a.tsv || a.format.is_some() {
        resolve_format(a.tsv, &a.format).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        })
    } else if let Some(out) = &a.output {
        match out.extension().and_then(|e| e.to_str()) {
            Some("tsv") => LineListFormat::Tsv,
            _ => LineListFormat::Parquet,
        }
    } else {
        LineListFormat::Parquet
    };

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

/// Parse a `--scheme` string. Phase 1: only `flat:RATE`.
fn parse_scheme(s: &str) -> Result<Box<dyn SamplingScheme>, String> {
    if let Some(rest) = s.strip_prefix("flat:") {
        let rate: f64 = rest
            .parse()
            .map_err(|e| format!("invalid flat sampling rate '{}': {}", rest, e))?;
        if !(0.0..=1.0).contains(&rate) {
            return Err(format!("flat sampling rate must be in [0, 1], got {}", rate));
        }
        Ok(Box::new(Flat::new(rate)))
    } else {
        Err(format!(
            "unknown sampling scheme '{}'. Phase 1 supports 'flat:RATE' \
             (e.g. flat:0.1).",
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
pub fn cmd_lineage_tree(a: &LineageTreeArgs) {
    let entries = read_line_list(&a.line_list).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let scheme = parse_scheme(&a.scheme).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let forest = TransmissionForest::from_entries(&entries);
    let mut rng = sim::rng::StatefulRng::new(a.seed);
    let sampled = scheme.select(&forest, &mut rng);

    if sampled.is_empty() {
        eprintln!(
            "warning: sampling scheme selected 0 tips from {} candidate leaves; \
             no tree to emit",
            forest.leaves().len()
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
        "lineage tree: {} candidate leaves, {} sampled tips, {} tree(s)",
        forest.leaves().len(),
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
