//! The view of one stored fit run that every reader of its files takes: what
//! it ran (its method and backend), its run id and its problem's, and the
//! problem's provenance, read from a copy of the run's files
//! ([`crate::fit_store::RunFiles`]).
//!
//! A run's folder holds its result files, `run_inputs.json` (what it was
//! asked to do and what it found) and `fit.meta.json` (the problem's
//! provenance as the run resolved it); the copy a reader opens also holds
//! `model.ir.json`, the stored compile the problem was posed on
//! ([`crate::fit_store::open_fit`]). [`FitView::read`] reads that directory
//! and nothing else.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::run_meta::{FitAlgorithm, InferenceBackend};

/// The stored run a view reads.
#[derive(Debug, Clone)]
pub struct FitMethodView {
    /// The directory holding the run's files.
    pub method_dir: PathBuf,
    /// Inference algorithm tag.
    pub method: FitAlgorithm,
    /// The run's id (64 hex digits): the address `camdl show` and `fit
    /// predict` accept.
    pub run_id: String,
    /// The run's label: a stored run's files hold none, so [`FitView::read`]
    /// leaves it `None` and the store's walker fills it from the invocation
    /// that named the run.
    pub label: Option<String>,
    /// When the run started; empty from [`FitView::read`], which reads the
    /// run's files alone, and filled by the store's walker from its record.
    pub created_at: String,
    /// Simulation backend the method ran on.
    pub backend: InferenceBackend,
}

/// The view of one stored fit run and its problem.
#[derive(Debug, Clone)]
pub struct FitView {
    /// The problem's run id: the fit id `fit table` and `camdl list` show.
    pub fit_hash: String,
    /// The directory the view was read from.
    pub segment: PathBuf,
    pub model_identity: String,
    /// The fit file the run was given, as typed; `None` from
    /// [`FitView::read`], and filled by the store's walker from the run's
    /// record.
    pub source_config: Option<String>,
    pub fit_toml_hash: String,
    pub data_hashes: HashMap<String, String>,
    /// The run, alone.
    pub methods: Vec<FitMethodView>,
}

impl FitMethodView {
    /// A stored run is a completed one.
    pub fn completed(&self) -> bool {
        true
    }
}

impl FitView {
    /// The view of the run whose files `dir` holds: `None` when `dir` holds
    /// no `run_inputs.json`, or one that names no method this camdl knows.
    pub fn read(dir: &Path) -> Option<FitView> {
        let inputs = crate::fit_store::read_run_inputs(dir).ok()?;
        let method: FitAlgorithm = match inputs.get("method").map(|v| serde_json::from_value(v.clone())) {
            Some(Ok(m)) => m,
            Some(Err(_)) | None => {
                crate::out::detached().say(format_args!(
                    "warning: skipping fit run {}: it records no method this camdl knows",
                    dir.display()
                ));
                return None;
            }
        };
        let backend: InferenceBackend = inputs
            .get("backend")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(InferenceBackend::ChainBinomial);
        let text = |k: &str| inputs.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let side = crate::run_meta::read_fit_sidecar(dir).ok().flatten();
        let run = FitMethodView {
            method_dir: dir.to_path_buf(),
            method,
            run_id: text("run_id"),
            label: None,
            created_at: String::new(),
            backend,
        };
        Some(FitView {
            fit_hash: text("problem_id"),
            segment: dir.to_path_buf(),
            model_identity: side.as_ref().map(|s| s.model_identity.clone()).unwrap_or_default(),
            source_config: None,
            fit_toml_hash: side.as_ref().map(|s| s.fit_toml_hash.clone()).unwrap_or_default(),
            data_hashes: side.map(|s| s.data_hashes.into_iter().collect()).unwrap_or_default(),
            methods: vec![run],
        })
    }
}
