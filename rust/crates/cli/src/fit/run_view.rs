//! One stored fit run as `camdl show` and `camdl list` present it: the run's
//! identity, the problem it belongs to, how its chains started, its stored
//! verdict and its recorded outputs.
//!
//! Everything here is read from the run's own `run.json`, its problem folder's
//! `fit.meta.json`, and the method's stored summary files. Nothing is
//! recomputed, and nothing here feeds a hash.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::cas_read::Leaf;

/// The schema tag of `camdl show --format json` on a fit run.
pub const FIT_SHOW_SCHEMA_NAME: &str = "fit_show";
pub const FIT_SHOW_SCHEMA_VERSION: u32 = 1;

/// The problem a run belongs to: its folder's fit id and the files it was
/// made from, relative to the folder (`None` when not recorded).
#[derive(Debug, Clone)]
pub struct Problem {
    pub fit_id: String,
    pub model: Option<String>,
    pub config: Option<String>,
}

impl Problem {
    pub fn of(segment: &Path) -> Problem {
        let view = crate::fit::fit_view::FitView::read(segment);
        Problem {
            fit_id: view.as_ref().map(|v| v.fit_hash.clone()).unwrap_or_default(),
            model: view.as_ref().and_then(|v| v.source_model.clone()),
            config: view.as_ref().and_then(|v| v.source_config.clone()),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({ "fit_id": self.fit_id, "model": self.model, "config": self.config })
    }

    /// `76aecea7 · models/m0.camdl · configs/m0.toml`.
    pub fn line(&self) -> String {
        let mut parts = vec![short(&self.fit_id, 8).to_string()];
        parts.extend(self.model.iter().cloned());
        parts.extend(self.config.iter().cloned());
        parts.join(" · ")
    }
}

/// How a run's chains started, as its record states it.
#[derive(Debug, Clone, PartialEq)]
pub struct Starts {
    /// The rule's tag (`from_prior`, `from_posterior`, …).
    pub rule: String,
    /// The source handle as typed, for a sourced rule.
    pub handle: Option<String>,
    /// The source run's id, when the source was a stored run.
    pub run_id: Option<String>,
    /// The source chains a `from_posterior` rule drew from.
    pub chains: Option<Vec<u64>>,
}

impl Starts {
    pub fn of(record: &runid::RunRecord) -> Starts {
        let inputs = &record.inputs;
        let spelled = inputs.get("starts").and_then(Value::as_str).unwrap_or("");
        let rule = spelled.split_whitespace().next().unwrap_or("—").to_string();
        let src = inputs.get("starts_source").filter(|v| !v.is_null());
        Starts {
            rule,
            handle: src.and_then(|s| s.get("handle")).and_then(Value::as_str).map(String::from),
            run_id: src.and_then(|s| s.get("run_id")).and_then(Value::as_str).map(String::from),
            chains: src.and_then(|s| s.get("chains")).and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect()),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({ "rule": self.rule, "handle": self.handle, "run_id": self.run_id, "chains": self.chains })
    }

    /// `from_posterior run 3a63b6 (@m0_mh) chains 0,1,2,4`: the recorded
    /// source run id first, the handle as typed beside it, since the handle
    /// may since have moved.
    pub fn cell(&self) -> String {
        let mut s = self.rule.clone();
        match (&self.run_id, &self.handle) {
            (Some(id), Some(h)) => s.push_str(&format!(" run {} ({h})", short(id, 6))),
            (Some(id), None) => s.push_str(&format!(" run {}", short(id, 6))),
            (None, Some(h)) => s.push_str(&format!(" {h}")),
            (None, None) => {}
        }
        if let Some(ch) = &self.chains {
            let ids: Vec<String> = ch.iter().map(u64::to_string).collect();
            s.push_str(&format!(" chains {}", ids.join(",")));
        }
        s
    }
}

/// A run's stored verdict: the line `fit summary` leads with.
#[derive(Debug, Clone, PartialEq)]
pub enum RunVerdict {
    /// A sampler's R̂ verdict, from its schema-tagged `<method>_summary.json`.
    Posterior { verdict: score::convergence::Verdict, min_ess_bulk: Option<f64> },
    /// IF2 and the optimizers: the gating verdict from `fit_state.toml`.
    Gate(String),
    /// Nothing to judge: one chain, or a method with no chains to compare.
    None,
    /// The run has no result to judge (it did not complete).
    NotCompleted,
    /// A stored summary that could not be read, with why.
    Unreadable(String),
}

impl RunVerdict {
    pub fn of(leaf: &Leaf) -> RunVerdict {
        use crate::fit::method_result::MethodResult;
        use score::convergence::GateVerdict;
        if leaf.record.status != runid::RunStatus::Completed {
            return RunVerdict::NotCompleted;
        }
        let method = leaf.level_label("method").to_string();
        let result = match MethodResult::load_from(&leaf.dir, &method) {
            Ok(r) => r,
            Err(e) => return RunVerdict::Unreadable(e.to_string()),
        };
        let gate = |g: GateVerdict| match g {
            GateVerdict::NotAssessed => RunVerdict::None,
            g => RunVerdict::Gate(
                serde_json::to_value(g).ok().and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_default(),
            ),
        };
        match &result {
            MethodResult::Pgas(r) => RunVerdict::Posterior {
                verdict: crate::fit::report::facts::verdict(&result),
                min_ess_bulk: r.diagnostics.min_ess(),
            },
            MethodResult::Pmmh(r) => RunVerdict::Posterior {
                verdict: crate::fit::report::facts::verdict(&result),
                min_ess_bulk: r.diagnostics.min_ess(),
            },
            MethodResult::If2(r) => gate(r.gate_verdict),
            MethodResult::Nlopt(_) => match crate::fit::state::FitState::load(&leaf.dir.to_string_lossy())
                .ok()
                .and_then(|s| s.optimizer_convergence)
            {
                Some(c) if c.gate_verdict() == "not_assessed" => RunVerdict::None,
                Some(c) => RunVerdict::Gate(c.gate_verdict().to_string()),
                None => RunVerdict::Unreadable("fit_state.toml records no optimizer verdict".into()),
            },
            MethodResult::Pfilter(_) => RunVerdict::None,
        }
    }

    /// `converged (R̂ 1.006)`, `not converged (R̂ 2.18)`, `gate pass`, `—`.
    pub fn cell(&self) -> String {
        use score::convergence::Verdict;
        match self {
            RunVerdict::Posterior { verdict, .. } => match verdict {
                Verdict::Converged { max_rhat, .. } => format!("converged (R̂ {})", rhat(*max_rhat)),
                Verdict::NotConverged { max_rhat, .. } => format!("not converged (R̂ {})", rhat(*max_rhat)),
                Verdict::ChainsFrozen { chains } => format!(
                    "not converged (chains {} frozen)",
                    chains.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
                ),
                Verdict::Unassessable { params } => {
                    format!("not converged (R̂ not computable for {})", params.join(", "))
                }
                Verdict::NotAssessed { reason } => format!("not assessed ({reason})"),
                Verdict::NotApplicable => "—".to_string(),
            },
            RunVerdict::Gate(g) => format!("gate {g}"),
            RunVerdict::None | RunVerdict::NotCompleted => "—".to_string(),
            RunVerdict::Unreadable(why) => format!("unreadable: {why}"),
        }
    }

    /// `{kind, max_rhat, min_ess_bulk}`; `kind` is the verdict's tag.
    pub fn to_json(&self) -> Value {
        use score::convergence::Verdict;
        match self {
            RunVerdict::Posterior { verdict, min_ess_bulk } => {
                let max_rhat = match verdict {
                    Verdict::Converged { max_rhat, .. } | Verdict::NotConverged { max_rhat, .. } => {
                        finite(*max_rhat)
                    }
                    _ => Value::Null,
                };
                let kind = serde_json::to_value(verdict)
                    .ok()
                    .and_then(|v| v.get("kind").cloned())
                    .unwrap_or(Value::Null);
                json!({ "kind": kind, "max_rhat": max_rhat,
                        "min_ess_bulk": min_ess_bulk.map(finite).unwrap_or(Value::Null) })
            }
            RunVerdict::Gate(g) => json!({ "kind": "gate", "gate": g, "max_rhat": null, "min_ess_bulk": null }),
            RunVerdict::None => json!({ "kind": "none", "max_rhat": null, "min_ess_bulk": null }),
            RunVerdict::NotCompleted => json!({ "kind": "not_completed", "max_rhat": null, "min_ess_bulk": null }),
            RunVerdict::Unreadable(why) => json!({ "kind": "unreadable", "reason": why, "max_rhat": null, "min_ess_bulk": null }),
        }
    }
}

fn finite(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// R̂ to three decimals below 2, two above, as `fit summary` prints it.
fn rhat(x: f64) -> String {
    if !x.is_finite() {
        "∞".to_string()
    } else if x < 2.0 {
        format!("{x:.3}")
    } else {
        format!("{x:.2}")
    }
}

/// The first `n` characters of a hex id.
pub fn short(id: &str, n: usize) -> &str {
    &id[..n.min(id.len())]
}

/// The live state of a run as `list` and `show` print it.
pub fn status_word(leaf: &Leaf) -> &'static str {
    match runid::store::leaf_state(&leaf.dir, &leaf.record) {
        runid::store::LeafState::Completed => "completed",
        runid::store::LeafState::Failed { .. } | runid::store::LeafState::FailedReasonUnknown => "failed",
        runid::store::LeafState::Running { .. } => "running",
        runid::store::LeafState::OwnerUnknown { .. } => "running elsewhere",
        runid::store::LeafState::Interrupted { .. } => "interrupted",
    }
}

/// One output a run holds: a path relative to the run's leaf (or, for a
/// prediction, to the problem folder) and what it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub path: String,
    pub role: String,
}

/// The run's outputs: its recorded artifact set (`run.json` `artifacts`), each
/// with the role its `output_schema` records, then its predictions under the
/// problem folder's `predict/<run id>…/`. Never a directory listing of the
/// leaf, so a stray file is not reported as an output.
pub fn outputs(leaf: &Leaf, segment: &Path) -> Vec<Output> {
    let schema_role = |path: &str| -> Option<String> {
        let key = per_chain_key(path);
        leaf.record
            .output_schema
            .get(&key)
            .and_then(|t| serde_json::to_value(t.role).ok())
            .and_then(|v| v.as_str().map(String::from))
    };
    let mut out: Vec<Output> = leaf
        .record
        .artifacts
        .keys()
        .map(|p| Output { path: p.clone(), role: schema_role(p).unwrap_or_else(|| "file".into()) })
        .collect();
    let run_id = leaf.record.run_id.to_hex().to_string();
    let bare = crate::fit::fit_view::predictions_dir(segment, &run_id, None);
    let stem = bare.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(segment.join("predict"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|d| {
            let n = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            d.is_dir() && (n == stem || n.starts_with(&format!("{stem}-")))
        })
        .collect();
    dirs.sort();
    for d in dirs {
        let mut files: Vec<PathBuf> = Vec::new();
        collect_files(&d, &mut files);
        files.sort();
        for f in files {
            let rel = f.strip_prefix(segment).unwrap_or(&f).to_string_lossy().into_owned();
            out.push(Output { path: rel, role: "prediction".into() });
        }
    }
    out
}

/// `chain_3/trace.tsv` → `chain_{n}/trace.tsv`, the key `output_schema` uses
/// for per-chain files.
fn per_chain_key(path: &str) -> String {
    match path.split_once('/') {
        Some((first, rest)) if score::convergence::ChainIndex::parse_dir_name(first).is_some() => {
            format!("chain_{{n}}/{rest}")
        }
        _ => path.to_string(),
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_files(&p, out);
        } else {
            out.push(p);
        }
    }
}
