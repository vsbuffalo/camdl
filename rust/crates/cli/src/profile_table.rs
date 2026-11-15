//! The collected table of a `camdl profile` run (gh#1059): one row per grid
//! cell, gathered from the cell's leaves once the run ends.
//!
//! Each `(cell × seed × start)` is its own content-addressed leaf holding an
//! `mle.toml` (`final_loglik`, `[focal]`, `[mle]`). The profile at a cell is
//! the best of those starts, so a row carries the swept values, the best
//! finite log-likelihood over every seed and start of the cell, the
//! estimated parameters at that start, the spread of the finite
//! log-likelihoods, how many starts completed out of how many ran, and the
//! best start's `run_id`. The table is read back from the leaves rather than
//! kept from the parallel loop, so a rerun served from the store prints the
//! same table. It is a view: nothing in it is hashed or stored.

use std::path::Path;

/// One start's result, read from its leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct StartOutcome {
    /// `final_loglik`; non-finite when the start failed.
    pub loglik: f64,
    /// The estimated parameters at this start's optimum, in table order.
    pub estimates: Vec<f64>,
    /// The leaf's `run_id`.
    pub run_id: String,
}

/// One grid cell: its swept values and every start that ran there, in
/// (seed, start) order. `None` is a start with no readable leaf.
#[derive(Debug, Clone, PartialEq)]
pub struct CellRow {
    pub focal: Vec<f64>,
    pub starts: Vec<Option<StartOutcome>>,
}

impl CellRow {
    fn finite(&self) -> impl Iterator<Item = &StartOutcome> {
        self.starts.iter().flatten().filter(|s| s.loglik.is_finite())
    }

    /// The start with the highest finite log-likelihood; the first such in
    /// (seed, start) order on a tie.
    pub fn best(&self) -> Option<&StartOutcome> {
        self.finite().fold(None, |best: Option<&StartOutcome>, s| match best {
            Some(b) if b.loglik >= s.loglik => Some(b),
            _ => Some(s),
        })
    }

    /// Starts that produced a finite log-likelihood.
    pub fn n_completed(&self) -> usize {
        self.finite().count()
    }

    /// `max − min` of the finite log-likelihoods; `None` when none is finite.
    pub fn spread(&self) -> Option<f64> {
        let (lo, hi) = self.finite().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.loglik), hi.max(s.loglik))
        });
        (lo <= hi).then_some(hi - lo)
    }
}

/// The whole table.
#[derive(Debug, Clone)]
pub struct ProfileTable {
    pub focal_names: Vec<String>,
    pub estimated: Vec<String>,
    /// Which likelihood `loglik` is (`ic_free` makes it conditional on y₁).
    pub objective: crate::fit::config_v2::LikelihoodObjective,
    /// The class of the log-likelihood (`if2`, `ode_marginal`, ...).
    pub loglik_type: &'static str,
    pub rows: Vec<CellRow>,
}

/// A TSV cell: shortest round-trip decimal, `NaN` for a missing value.
fn tsv_num(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x}"),
        None => "NaN".to_string(),
    }
}

impl ProfileTable {
    /// One header line, one line per cell. A cell with no finite start has
    /// `NaN` for its log-likelihood, estimates and spread, and an empty
    /// `run_id`.
    pub fn to_tsv(&self) -> String {
        let mut header: Vec<&str> = self.focal_names.iter().map(String::as_str).collect();
        header.push("loglik");
        header.extend(self.estimated.iter().map(String::as_str));
        header.extend(["loglik_spread_starts", "starts_n_completed", "n_starts", "run_id"]);
        let mut out = header.join("\t");
        out.push('\n');
        for row in &self.rows {
            let best = row.best();
            let mut fields: Vec<String> = row.focal.iter().map(|v| tsv_num(Some(*v))).collect();
            fields.push(tsv_num(best.map(|b| b.loglik)));
            for i in 0..self.estimated.len() {
                fields.push(tsv_num(best.map(|b| b.estimates[i])));
            }
            fields.push(tsv_num(row.spread()));
            fields.push(row.n_completed().to_string());
            fields.push(row.starts.len().to_string());
            fields.push(best.map(|b| b.run_id.clone()).unwrap_or_default());
            out.push_str(&fields.join("\t"));
            out.push('\n');
        }
        out
    }

    /// The same rows as a JSON document; a missing value is `null`.
    pub fn to_json(&self) -> serde_json::Value {
        let cells: Vec<serde_json::Value> = self.rows.iter().map(|row| {
            let best = row.best();
            let focal: serde_json::Map<String, serde_json::Value> = self.focal_names.iter()
                .zip(&row.focal)
                .map(|(n, v)| (n.clone(), serde_json::json!(v)))
                .collect();
            let estimates: serde_json::Map<String, serde_json::Value> = self.estimated.iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), serde_json::json!(best.map(|b| b.estimates[i]))))
                .collect();
            serde_json::json!({
                "focal": focal,
                "loglik": best.map(|b| b.loglik),
                "estimates": estimates,
                "loglik_spread_starts": row.spread(),
                "starts_n_completed": row.n_completed(),
                "n_starts": row.starts.len(),
                "run_id": best.map(|b| b.run_id.clone()),
            })
        }).collect();
        serde_json::json!({
            "focal": self.focal_names,
            "estimated": self.estimated,
            "likelihood_objective": self.objective,
            "loglik_type": self.loglik_type,
            "cells": cells,
        })
    }
}

/// Read one start's `mle.toml` from its leaf directory: `final_loglik` and
/// the `[mle]` value of each name in `estimated`.
pub fn read_start(leaf: &Path, estimated: &[String], run_id: String) -> Result<StartOutcome, String> {
    let path = leaf.join("mle.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let table: toml::Table = text.parse()
        .map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
    // `render_mle_toml` writes an integral float as a bare integer.
    let num = |v: &toml::Value| v.as_float().or_else(|| v.as_integer().map(|i| i as f64));
    let loglik = table.get("final_loglik").and_then(num)
        .ok_or_else(|| format!("{} has no numeric final_loglik", path.display()))?;
    let mle = table.get("mle").and_then(|m| m.as_table())
        .ok_or_else(|| format!("{} has no [mle] table", path.display()))?;
    let estimates = estimated.iter()
        .map(|n| mle.get(n).and_then(num)
            .ok_or_else(|| format!("{} [mle] has no value for {n}", path.display())))
        .collect::<Result<Vec<f64>, String>>()?;
    Ok(StartOutcome { loglik, estimates, run_id })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(loglik: f64, beta: f64, id: &str) -> Option<StartOutcome> {
        Some(StartOutcome { loglik, estimates: vec![beta], run_id: id.into() })
    }

    fn table(rows: Vec<CellRow>) -> ProfileTable {
        ProfileTable {
            focal_names: vec!["gamma".into()],
            estimated: vec!["beta".into()],
            objective: crate::fit::config_v2::LikelihoodObjective::Full,
            loglik_type: "ode_marginal",
            rows,
        }
    }

    /// The best start is the highest finite log-likelihood; a failed start
    /// (non-finite, or no leaf) counts as run but not completed and never
    /// enters the spread.
    #[test]
    fn a_row_is_its_best_finite_start() {
        let row = CellRow {
            focal: vec![0.12],
            starts: vec![start(-30.0, 0.31, "a"), start(f64::NEG_INFINITY, 0.9, "b"),
                         None, start(-29.5, 0.33, "c")],
        };
        assert_eq!(row.best().unwrap().run_id, "c");
        assert_eq!(row.n_completed(), 2);
        assert_eq!(row.spread(), Some(0.5));
        assert_eq!(table(vec![row]).to_tsv(),
            "gamma\tloglik\tbeta\tloglik_spread_starts\tstarts_n_completed\tn_starts\trun_id\n\
             0.12\t-29.5\t0.33\t0.5\t2\t4\tc\n");
    }

    /// A cell where no start completed is kept, with nothing to report: the
    /// row says so instead of dropping the grid point.
    #[test]
    fn a_cell_with_no_finite_start_is_a_row_of_nan() {
        let row = CellRow { focal: vec![0.15], starts: vec![start(f64::NEG_INFINITY, 0.9, "b"), None] };
        let t = table(vec![row]);
        assert_eq!(t.to_tsv().lines().nth(1).unwrap(), "0.15\tNaN\tNaN\tNaN\t0\t2\t");
        let cell = &t.to_json()["cells"][0];
        assert!(cell["loglik"].is_null() && cell["estimates"]["beta"].is_null());
        assert_eq!(cell["starts_n_completed"], 0);
    }

    #[test]
    fn read_start_takes_an_integral_value_written_bare() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("mle.toml"),
            "final_loglik = -11\n\n[focal]\nR0 = 25\n\n[mle]\nbeta = 1\n").unwrap();
        let s = read_start(dir.path(), &["beta".into()], "id".into()).unwrap();
        assert_eq!((s.loglik, s.estimates), (-11.0, vec![1.0]));
        assert!(read_start(dir.path(), &["gamma".into()], "id".into())
            .unwrap_err().contains("no value for gamma"));
    }
}
