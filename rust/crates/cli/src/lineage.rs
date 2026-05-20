//! CLI glue for the individual-sampling (lineage) layer.
//!
//! Two entry points:
//!   - [`run_simulate_lineages`]: the `camdl simulate --lineages` path. Picks
//!     the line-list format (Parquet default, TSV via `--tsv` / `--format
//!     tsv`), runs Gillespie with the lineage observer attached, and writes the
//!     count trajectory (unless suppressed) plus the streamed line list.
//!   - [`cmd_lineage_tree`]: the offline `camdl lineage tree` projection. Pure
//!     function over a line-list file → sampled transmission tree → Newick.

use std::io::Write;
use std::path::{Path, PathBuf};

use sim::lineage::{
    tree::{Flat, SamplingScheme, TransmissionForest},
    LineListFormat, LineListWriter, TsvLineListWriter,
};

use crate::args::{LineageTreeArgs, SimulateArgs};
use crate::util::SimRun;

/// Resolve the requested line-list format from `--format` / `--tsv`.
/// Default is Parquet (production), per the proposal.
fn resolve_format(a: &SimulateArgs) -> Result<LineListFormat, String> {
    if a.tsv {
        return Ok(LineListFormat::Tsv);
    }
    match &a.format {
        None => Ok(LineListFormat::Parquet),
        Some(s) => LineListFormat::parse(s)
            .ok_or_else(|| format!("unknown --format '{}'; expected 'parquet' or 'tsv'", s)),
    }
}

/// Default output path for a given format, when `--lineage-out` is absent.
fn default_out(format: LineListFormat) -> PathBuf {
    match format {
        LineListFormat::Tsv => PathBuf::from("line_list.tsv"),
        LineListFormat::Parquet => PathBuf::from("line_list.parquet"),
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

/// `camdl simulate --lineages` — run Gillespie with lineage tracking.
pub fn run_simulate_lineages(a: &SimulateArgs, run: &SimRun) {
    let format = resolve_format(a).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let out_path = a
        .lineage_out
        .clone()
        .unwrap_or_else(|| default_out(format));

    let writer = build_writer(format, &out_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let (traj, model, diag) = crate::util::run_simulation_lineage(run, writer).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    eprintln!(
        "lineage tracking: wrote line list to {} ({})",
        out_path.display(),
        match format {
            LineListFormat::Tsv => "tsv",
            LineListFormat::Parquet => "parquet",
        }
    );

    // Surface the sub-dt bias diagnostic. Gillespie is exact (fraction 0);
    // tau-leap / chain-binomial report the edge-weighted fraction of
    // transmission edges whose sub-dt ordering the frozen-pool approximation
    // could not resolve. A non-trivial fraction is a signal to shrink dt or use
    // Gillespie for trustworthy benchmark trees.
    if diag.exact {
        eprintln!(
            "lineage sub-dt bias: 0.000 (exact — Gillespie; {} transmission edges)",
            diag.edges
        );
    } else {
        eprintln!(
            "lineage sub-dt bias: {:.3} ({} transmission edges; frozen-pool \
             approximation — shrink --dt or use --backend gillespie for \
             trustworthy trees)",
            diag.fraction, diag.edges
        );
    }

    // Count trajectory output (stdout or --output). The trajectory is
    // byte-identical to a run without --lineages at the same seed.
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
