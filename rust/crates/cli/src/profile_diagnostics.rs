//! gh#74 Option B: per-cell convergence diagnostics for `camdl profile`.
//!
//! Each profile grid cell runs K independent `--n-starts` optimizations
//! under one of two algorithm families (IF2, NLopt). The diagnostic
//! data each start produces — per-iteration loglik traces and
//! cooling-schedule end-states — is kept so a user (or coding agent)
//! can tell whether a jagged profile is Monte Carlo noise that more
//! compute would smooth, genuine multimodality, or per-cell
//! optimization failures.
//!
//! This module owns the per-start → per-cell aggregation:
//!
//! * `PerStartDiagnostics` is the per-start record. The profile driver
//!   populates one per (cell × start) and serializes it into that
//!   start's `mle.toml` under a `[diagnostics]` block. Persisting it
//!   on disk (rather than threading it through the parallel loop) is
//!   what lets the per-cell rollup phase — which already scans every
//!   start's `mle.toml` — compute aggregates without changing the
//!   job-scheduling shape.
//!
//! * `CellDiagnostics` is the per-cell aggregate. The rollup reads
//!   each `mle.toml`'s `[diagnostics]` block, builds the per-cell
//!   diagnostic, and emits it as new columns in the per-seed
//!   `profile.tsv` and the umbrella `summary.tsv`.
//!
//! The aggregate columns are documented in `docs/inference.md`
//! ("Per-cell diagnostics") and in the `camdl profile --help` OUTPUT
//! section. Schema is stable per-run; algorithms that don't supply a
//! given column write `NaN` (capital N — matches camdl's existing TSV
//! NaN convention).

use serde::{Deserialize, Serialize};

/// Algorithm tag carried per-start. Determines which fields of
/// `PerStartDiagnostics` are populated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagAlgo {
    /// IF2 — exposes `iterations_used`, `cooling_final`, and the per-iter
    /// (perturbed) loglik trace.
    If2,
    /// NLopt deterministic optimizer — exposes only the final loglik
    /// (no trace, no acceptance rate, no cooling).
    Nlopt,
}

impl DiagAlgo {
    pub fn as_str(self) -> &'static str {
        match self {
            DiagAlgo::If2   => "if2",
            DiagAlgo::Nlopt => "nlopt",
        }
    }
}

/// Per-start convergence record. Persisted into each `mle.toml`'s
/// `[diagnostics]` block; the rollup parses it back and aggregates
/// across the K starts at one cell.
#[derive(Clone, Debug, Default)]
pub struct PerStartDiagnostics {
    /// Which algorithm produced this start. Drives the aggregate
    /// column population at rollup time.
    pub algo: Option<DiagAlgo>,
    /// Whether the start ran to completion. False when the inference
    /// engine returned `Err` (treated as -Inf loglik by the profile
    /// driver) or when `final_loglik` is non-finite.
    pub completed: bool,
    /// Final cooling-step index reached (IF2 only).
    pub iterations_used: Option<usize>,
    /// Final perturbation SD applied at the last cooling step (IF2
    /// only). Mean across estimated params at the final iteration —
    /// scalar summary of "how cool we got."
    pub cooling_final: Option<f64>,
    /// Per-iteration (IF2) loglik trace. Used by
    /// the rollup to compute Gelman-Rubin Rhat across the K starts.
    /// Empty for NLopt (no trace surface). For IF2 the trace carries
    /// `if2_perturbed_loglik` — a diagnostic, not the model loglik —
    /// matching how the IF2 engine docs describe it; Rhat is a chain-
    /// level agreement metric, so the perturbed trace is a fine
    /// proxy for "are the starts wandering the same basin."
    pub loglik_trace: Vec<f64>,
}

impl PerStartDiagnostics {
    /// Render to a TOML fragment for embedding inside the per-start
    /// `mle.toml`. Caller writes the result under a `[diagnostics]`
    /// section header. Trace arrays use the inline-array form so
    /// they survive a round-trip through any TOML 1.0 parser.
    pub fn to_toml_fragment(&self) -> String {
        let mut body = String::new();
        if let Some(algo) = self.algo {
            body.push_str(&format!("algorithm = \"{}\"\n", algo.as_str()));
        }
        body.push_str(&format!("completed = {}\n", self.completed));
        if let Some(v) = self.iterations_used {
            body.push_str(&format!("iterations_used = {}\n", v));
        }
        if let Some(v) = self.cooling_final {
            body.push_str(&format!("cooling_final = {}\n", v));
        }
        write_f64_trace(&mut body, "loglik_trace", &self.loglik_trace);
        body
    }
}

/// Emit `<name> = [v0, v1, ...]` to `body` when `trace` is non-empty.
/// Skips entirely when empty so NLopt `mle.toml` files stay
/// tight (no spurious empty arrays). NaN/±Inf serialised as TOML 1.0
/// reserved keywords (`nan`, `inf`, `-inf`).
fn write_f64_trace(body: &mut String, name: &str, trace: &[f64]) {
    if trace.is_empty() { return; }
    body.push_str(name);
    body.push_str(" = [");
    for (i, v) in trace.iter().enumerate() {
        if i > 0 { body.push_str(", "); }
        if v.is_finite() {
            body.push_str(&format!("{}", v));
        } else if v.is_nan() {
            body.push_str("nan");
        } else if *v == f64::INFINITY {
            body.push_str("inf");
        } else {
            body.push_str("-inf");
        }
    }
    body.push_str("]\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-start `[diagnostics]` block is what each CAS leaf's
    /// `mle.toml` carries; the M4 reindex (gh#154) parses it back to
    /// rebuild the per-cell rollup. Pin that the live serializer emits
    /// the algorithm tag, the completion flag, the IF2 cooling end-state
    /// and the trace array as a TOML-parseable fragment.
    #[test]
    fn to_toml_fragment_emits_parseable_diagnostics() {
        let d = PerStartDiagnostics {
            algo: Some(DiagAlgo::If2),
            completed: true,
            iterations_used: Some(3),
            cooling_final: Some(0.25),
            loglik_trace: vec![-10.0, -9.5, -9.0],
        };
        let body = format!("[diagnostics]\n{}", d.to_toml_fragment());
        let doc: toml::Value =
            toml::from_str(&body).expect("diagnostics fragment must be valid TOML");
        let t = doc.get("diagnostics").and_then(|v| v.as_table()).expect("[diagnostics] table");
        assert_eq!(t.get("algorithm").and_then(|v| v.as_str()), Some("if2"));
        assert_eq!(t.get("completed").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(t.get("iterations_used").and_then(|v| v.as_integer()), Some(3));
        assert_eq!(t.get("cooling_final").and_then(|v| v.as_float()), Some(0.25));
        // `write_f64_trace` emits integer-valued floats as bare ints
        // (`-10`, not `-10.0`), which TOML parses back as Integer, not
        // Float. A correct parser of this fragment (the M4 reindex,
        // gh#154) must accept both — mirror that here.
        let trace: Vec<f64> = t
            .get("loglik_trace")
            .and_then(|v| v.as_array())
            .expect("loglik_trace array")
            .iter()
            .filter_map(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .collect();
        assert_eq!(trace, vec![-10.0, -9.5, -9.0]);
    }
}


