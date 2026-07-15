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

/// Classify one column name into its semantic role. `params` is every model
/// parameter name; `estimated` is the sampled subset. A column that is neither a
/// chain key, an iteration axis, nor a model parameter is a sampler diagnostic
/// (`loglik`, `log_posterior`, `tree_depth`, `accepted`, …).
fn classify(name: &str, params: &HashSet<&str>, estimated: &HashSet<&str>) -> ColumnRole {
    match name {
        "chain" => ColumnRole::Chain,
        "sweep" | "step" | "draw" | "iteration" => ColumnRole::Iteration,
        n if estimated.contains(n) => ColumnRole::ParamEstimated,
        n if params.contains(n) => ColumnRole::ParamFixed,
        _ => ColumnRole::Diagnostic,
    }
}

/// The tab-separated column names of `path`'s header, skipping a leading
/// `# <version>` comment line if present. `None` when the file is absent or
/// unreadable — the schema is best-effort provenance, never a hard dependency.
fn header(path: &Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| !l.starts_with('#'))?;
    Some(line.split('\t').map(str::to_string).collect())
}

fn table(cols: &[String], role: TableRole, params: &HashSet<&str>, estimated: &HashSet<&str>) -> TableSchema {
    TableSchema {
        role,
        columns: cols
            .iter()
            .map(|name| ColumnSpec { name: name.clone(), role: classify(name, params, estimated) })
            .collect(),
    }
}

/// The lowest-numbered `chain_*` directory under `leaf`, if any — the per-chain
/// trace schema is identical across chains, so one is read and keyed by `{n}`.
fn first_chain_dir(leaf: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(leaf)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("chain_"))
        })
        .collect();
    dirs.sort();
    dirs.into_iter().next()
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

    // draws.tsv — the thinned posterior cloud (Bayesian methods; if2 writes none).
    if let Some(cols) = header(&leaf.join("draws.tsv")) {
        out.insert("draws.tsv".to_string(), table(&cols, TableRole::PosteriorCloud, all_params, estimated));
    }

    // Per-chain trace — one entry keyed by the `{n}` wildcard, read from the
    // first chain directory. `trace.tsv` (pgas/pmmh/mh/nuts) or
    // `parameter_traces.tsv` (if2).
    if let Some(chain1) = first_chain_dir(leaf) {
        for fname in ["trace.tsv", "parameter_traces.tsv"] {
            if let Some(cols) = header(&chain1.join(fname)) {
                out.insert(format!("chain_{{n}}/{fname}"), table(&cols, TableRole::Trace, all_params, estimated));
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
    fn classify_axes_params_and_diagnostics() {
        let (params, est) = sets(&["beta", "gamma", "N0"], &["beta"]);
        assert_eq!(classify("chain", &params, &est), ColumnRole::Chain);
        assert_eq!(classify("sweep", &params, &est), ColumnRole::Iteration);
        assert_eq!(classify("draw", &params, &est), ColumnRole::Iteration);
        assert_eq!(classify("beta", &params, &est), ColumnRole::ParamEstimated);
        assert_eq!(classify("gamma", &params, &est), ColumnRole::ParamFixed);
        assert_eq!(classify("N0", &params, &est), ColumnRole::ParamFixed);
        // neither axis nor parameter → diagnostic
        assert_eq!(classify("log_posterior", &params, &est), ColumnRole::Diagnostic);
        assert_eq!(classify("n_divergent", &params, &est), ColumnRole::Diagnostic);
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
