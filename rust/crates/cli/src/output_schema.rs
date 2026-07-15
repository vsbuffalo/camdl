//! Column schema for a run's tabular outputs — the `run.json` `output_schema`
//! (proposal `docs/dev/proposals/2026-07-15-run-output-column-schema.md`).
//!
//! Rather than reconstruct a writer's column order (which would drift from the
//! writer), we read each written file's ACTUAL header and classify every column
//! by role against the model's parameter set. The declaration therefore cannot
//! disagree with the file it describes — the file is the single source of truth,
//! and classification needs only set membership, never column order.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use runid::record::{ColumnRole, ColumnSpec, TableRole, TableSchema};

/// Classify one column name into its semantic role, using the model's parameter
/// sets and a table-specific `default` for columns matching no reserved name or
/// prefix. `params` is every model parameter name; `estimated` is the sampled
/// subset.
///
/// The reserved vocabulary (`chain`, the iteration/time axes, `flow_`/`inc_`
/// prefixes) is assumed not to collide with model parameter names — safe because
/// camdl parameters are epidemiological (`beta`, `gamma`), never sampler or
/// trajectory column names. A colliding parameter would take its reserved role
/// rather than a parameter role, but that is unreachable for a real model.
fn classify(
    name: &str,
    params: &HashSet<&str>,
    estimated: &HashSet<&str>,
    default: ColumnRole,
) -> ColumnRole {
    match name {
        "chain" => ColumnRole::Chain,
        "replicate" => ColumnRole::Replicate,
        "scenario" => ColumnRole::Scenario,
        "t" | "time" | "date" => ColumnRole::Time,
        "sweep" | "step" | "draw" | "iteration" | "point_id" => ColumnRole::Iteration,
        n if n.starts_with("flow_") => ColumnRole::Flow,
        n if n.starts_with("inc_") => ColumnRole::Incidence,
        n if estimated.contains(n) => ColumnRole::ParamEstimated,
        n if params.contains(n) => ColumnRole::ParamFixed,
        _ => default,
    }
}

/// Column names of a TSV header — the first non-`#` line, tab-split. `None` when
/// there is no such line.
fn header_cols(text: &str) -> Option<Vec<String>> {
    let line = text.lines().find(|l| !l.starts_with('#'))?;
    Some(line.split('\t').map(str::to_string).collect())
}

/// The header columns of the file at `path` — best-effort (`None` when absent or
/// unreadable; the schema is provenance, never a hard dependency).
fn header(path: &Path) -> Option<Vec<String>> {
    header_cols(&std::fs::read_to_string(path).ok()?)
}

fn table(
    cols: &[String],
    role: TableRole,
    params: &HashSet<&str>,
    estimated: &HashSet<&str>,
    default: ColumnRole,
) -> TableSchema {
    TableSchema {
        role,
        columns: cols
            .iter()
            .map(|name| ColumnSpec { name: name.clone(), role: classify(name, params, estimated, default) })
            .collect(),
    }
}

/// The header of `fname` in the first `chain_*` directory that has a readable
/// one. The per-chain schema is identical across chains, so any chain suffices —
/// reading the first *readable* one (not merely the lexicographically-first
/// directory) keeps the entry present when an early chain crashed or is absent.
fn first_chain_header(leaf: &Path, fname: &str) -> Option<Vec<String>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(leaf)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("chain_"))
        })
        .collect();
    dirs.sort();
    dirs.iter().find_map(|d| header(&d.join(fname)))
}

/// Build the `output_schema` for a completed fit stage by classifying the actual
/// headers of the files it wrote under `leaf`. `estimated` is the sampled
/// parameter names; `all_params` is every model parameter (estimated ∪ fixed).
/// Best-effort: files that cannot be read are omitted.
pub fn fit_output_schema(
    leaf: &Path,
    all_params: &HashSet<&str>,
    estimated: &HashSet<&str>,
) -> BTreeMap<String, TableSchema> {
    let mut out = BTreeMap::new();
    let diag = ColumnRole::Diagnostic;

    // draws.tsv — the thinned posterior cloud (Bayesian methods; if2 writes none).
    if let Some(cols) = header(&leaf.join("draws.tsv")) {
        out.insert(
            "draws.tsv".to_string(),
            table(&cols, TableRole::PosteriorCloud, all_params, estimated, diag),
        );
    }

    // Per-chain trace — one entry per trace filename, keyed by `{n}`, read from
    // the first chain that has it. `trace.tsv` (pgas/pmmh/mh/nuts) or
    // `parameter_traces.tsv` (if2 / nlopt).
    for fname in ["trace.tsv", "parameter_traces.tsv"] {
        if let Some(cols) = first_chain_header(leaf, fname) {
            out.insert(
                format!("chain_{{n}}/{fname}"),
                table(&cols, TableRole::Trace, all_params, estimated, diag),
            );
        }
    }
    out
}

/// Build the `output_schema` for a simulation leaf by classifying its in-memory
/// artifact headers (the sim commits atomically, so the files are not yet on
/// disk). `traj.tsv`/`ensemble.tsv` are trajectories — a column matching no
/// reserved name is a compartment `state`; other artifacts are skipped for now.
pub fn sim_output_schema(files: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, TableSchema> {
    let none: HashSet<&str> = HashSet::new();
    let mut out = BTreeMap::new();
    for (name, bytes) in files {
        let role_default = match name.as_str() {
            "traj.tsv" | "ensemble.tsv" => Some((TableRole::Trajectory, ColumnRole::State)),
            _ => None,
        };
        if let Some((role, default)) = role_default {
            if let Ok(text) = std::str::from_utf8(bytes) {
                if let Some(cols) = header_cols(text) {
                    out.insert(name.clone(), table(&cols, role, &none, &none, default));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sets<'a>(all: &'a [&str], est: &'a [&str]) -> (HashSet<&'a str>, HashSet<&'a str>) {
        (all.iter().copied().collect(), est.iter().copied().collect())
    }

    #[test]
    fn classify_axes_params_and_defaults() {
        let (params, est) = sets(&["beta", "gamma", "N0"], &["beta"]);
        let diag = ColumnRole::Diagnostic;
        // fit columns (default = diagnostic)
        assert_eq!(classify("chain", &params, &est, diag), ColumnRole::Chain);
        assert_eq!(classify("sweep", &params, &est, diag), ColumnRole::Iteration);
        assert_eq!(classify("draw", &params, &est, diag), ColumnRole::Iteration);
        assert_eq!(classify("beta", &params, &est, diag), ColumnRole::ParamEstimated);
        assert_eq!(classify("gamma", &params, &est, diag), ColumnRole::ParamFixed);
        assert_eq!(classify("N0", &params, &est, diag), ColumnRole::ParamFixed);
        assert_eq!(classify("log_posterior", &params, &est, diag), ColumnRole::Diagnostic);
        // trajectory columns (no params; default = state)
        let none: HashSet<&str> = HashSet::new();
        let st = ColumnRole::State;
        assert_eq!(classify("t", &none, &none, st), ColumnRole::Time);
        assert_eq!(classify("date", &none, &none, st), ColumnRole::Time);
        assert_eq!(classify("flow_infection", &none, &none, st), ColumnRole::Flow);
        assert_eq!(classify("inc_cases", &none, &none, st), ColumnRole::Incidence);
        assert_eq!(classify("replicate", &none, &none, st), ColumnRole::Replicate);
        assert_eq!(classify("S", &none, &none, st), ColumnRole::State); // compartment → default
    }

    #[test]
    fn sim_schema_classifies_trajectory_header() {
        let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        files.insert("traj.tsv".to_string(), b"t\tS\tI\tR\tflow_infection\n0\t99\t1\t0\t0\n".to_vec());
        files.insert("event_log.tsv".to_string(), b"anything\n".to_vec()); // not a trajectory
        let schema = sim_output_schema(&files);
        assert!(!schema.contains_key("event_log.tsv"), "non-trajectory artifact skipped");
        let traj = &schema["traj.tsv"];
        assert_eq!(traj.role, TableRole::Trajectory);
        assert_eq!(traj.columns[0].role, ColumnRole::Time); // t — the x-axis
        assert_eq!(traj.columns[1].role, ColumnRole::State); // S (compartment)
        assert_eq!(traj.columns[4].role, ColumnRole::Flow); // flow_infection
    }

    #[test]
    fn schema_reads_real_headers_and_tags_the_index() {
        let dir = tempfile::tempdir().unwrap();
        let leaf = dir.path();
        // A pgas draws.tsv: chain, draw, estimated (beta), fixed (gamma).
        std::fs::write(leaf.join("draws.tsv"), "chain\tdraw\tbeta\tgamma\n0\t5\t0.4\t0.1\n").unwrap();
        // A pgas trace with the diagnostic prefix, keyed by {n}.
        std::fs::create_dir(leaf.join("chain_1")).unwrap();
        std::fs::write(
            leaf.join("chain_1/trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\tn_divergent\tbeta\n1\t-2\t-3\t0\t0.4\n",
        )
        .unwrap();

        let (params, est) = sets(&["beta", "gamma"], &["beta"]);
        let schema = fit_output_schema(leaf, &params, &est);

        let draws = &schema["draws.tsv"];
        assert_eq!(draws.role, TableRole::PosteriorCloud);
        assert_eq!(draws.columns[0].role, ColumnRole::Chain);
        assert_eq!(draws.columns[1].role, ColumnRole::Iteration); // `draw` — the x-axis
        assert_eq!(draws.columns[2].role, ColumnRole::ParamEstimated);
        assert_eq!(draws.columns[3].role, ColumnRole::ParamFixed);

        let trace = &schema["chain_{n}/trace.tsv"];
        assert_eq!(trace.role, TableRole::Trace);
        assert_eq!(trace.columns[0].role, ColumnRole::Iteration); // `sweep` — the x-axis
        assert_eq!(trace.columns[1].role, ColumnRole::Diagnostic);
        assert_eq!(trace.columns[2].role, ColumnRole::Diagnostic);
        assert_eq!(trace.columns[4].role, ColumnRole::ParamEstimated);
    }
}
