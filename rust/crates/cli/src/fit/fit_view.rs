//! New-format fit reader: project a CAS fit segment into the fit-level and
//! per-stage views every fit consumer needs.
//!
//! A content-addressed fit is a segment directory `fits/{stem}-{h8}/` holding
//! one `FitStage` `runid::RunRecord` leaf per (cell × stage) plus a single
//! fit-level provenance sidecar (`fit.meta.json`). There is no fit-wide
//! `run.json`: the fit-level identity is the `fit`-level hash shared by every
//! leaf, and the fit-wide attributes the leaves don't carry (user `--label`,
//! the gh#75 prior sources, `estimated`/`fixed`/`data_hashes`/`model_identity`)
//! live once on the sidecar.
//!
//! [`FitView::read`] folds those leaves + sidecar into one fit-level view; each
//! [`FitStageView`] carries the per-stage headline numbers (`method`, `seed`,
//! `n_chains`, `best_loglik`, `best_chain`, path-derived axes) read directly
//! from the leaf's `RunRecord`. This is the single consumer-facing projection
//! of the new format — `walk_fits_root`, `table_row`, `fit summary`, `browse`
//! all read it instead of synthesizing a legacy `run_meta::Run`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use runid::{ArtifactKind, RunRecord};

use crate::fit::fit_tree::{derive_axes, StageAxes};
use crate::run_meta::{InferenceBackend, FitAlgorithm, ParameterProvenance, ResolvedPriorEntry};

/// One fit-stage leaf, projected to the headline numbers the fit consumers
/// read directly from the leaf's `RunRecord` (`inputs` + path-derived `axes`).
#[derive(Debug, Clone)]
pub struct FitStageView {
    /// Stage directory holding this leaf's `run.json` (the consumer loads the
    /// typed θ̂ / diagnostics from sibling `fit_state.toml` etc. itself).
    pub stage_dir: PathBuf,
    /// Bare stage name (`scout`, `refine`, `pgas`), from the `inputs.stage`.
    pub stage: String,
    /// Inference algorithm tag.
    pub method: FitAlgorithm,
    /// Simulation backend the stage ran on. Defaults to `ChainBinomial` when
    /// absent.
    pub backend: InferenceBackend,
    pub seed: u64,
    pub n_chains: usize,
    pub best_loglik: Option<f64>,
    pub best_chain: Option<usize>,
    /// Path-derived `(data_kind, fit_seed, sweep_slug)` axes — `None` for a
    /// stage outside the canonical `{real,synthetic}/...` layout.
    pub axes: Option<StageAxes>,
}

/// Fit-level view of a CAS fit segment: the fit-wide identity + provenance plus
/// one [`FitStageView`] per stage leaf, folded from the segment's stage-leaf
/// `RunRecord`s and the fit-level `FitSidecar`.
#[derive(Debug, Clone)]
pub struct FitView {
    // ── identity / provenance (some from leaves, some from the sidecar) ──
    /// The `fit`-level (FitDigest) hash shared by every leaf; the
    /// `camdl show`/`label` address for the fit. The sidecar is never
    /// identity-bearing.
    pub fit_hash: String,
    /// camdl engine version (first leaf).
    pub engine_version: String,
    /// Latest leaf `created_at` (ISO 8601 UTC).
    pub created_at: String,
    /// argv that produced the fit (first leaf).
    pub argv: Vec<String>,
    /// User `--label` (sidecar).
    pub label: Option<String>,

    // ── sidecar provenance (never defaulted on a well-formed segment) ──
    pub model: String,
    pub model_identity: String,
    pub fit_toml_path: String,
    pub fit_toml_hash: String,
    pub data_hashes: HashMap<String, String>,
    pub estimated: Vec<String>,
    pub fixed: HashMap<String, f64>,
    pub resolved_priors: Vec<ResolvedPriorEntry>,
    pub parameters_provenance: HashMap<String, ParameterProvenance>,

    // ── from the leaves ──
    /// Bare stage names in execution order (the `NN-` ordinal prefix sorts
    /// topologically), deduplicated.
    pub stages_declared: Vec<String>,
    /// One view per discovered stage leaf, sorted by stage label.
    pub stages: Vec<FitStageView>,
}

/// The `stage` level's readable label (`"01-scout"`); `""` when absent.
fn stage_label(r: &RunRecord) -> String {
    r.levels
        .iter()
        .find(|l| l.name == "stage")
        .map(|l| l.label.clone())
        .unwrap_or_default()
}

/// The bare stage name from a `NN-stage` provenance label (`"01-scout"` →
/// `"scout"`); a label without an ordinal prefix is returned unchanged. Splits
/// on the first `-` only, so stage names containing `-` survive.
fn bare_stage_name(stage_label: &str) -> String {
    stage_label
        .split_once('-')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| stage_label.to_string())
}

/// Build a [`FitStageView`] from a fit-stage leaf record + its directory.
/// Returns `None` when the record is not a `FitStage` or its `inputs` lacks a
/// parseable `method`.
fn stage_view_from_record(seg: &Path, dir: &Path, rec: &RunRecord) -> Option<FitStageView> {
    if rec.kind != ArtifactKind::FitStage {
        return None;
    }
    let inputs = rec.inputs.as_object()?;
    let method: FitAlgorithm = inputs
        .get("method")
        .and_then(|v| serde_json::from_value(v.clone()).ok())?;
    let backend: InferenceBackend = inputs
        .get("backend")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(InferenceBackend::ChainBinomial);
    Some(FitStageView {
        stage_dir: dir.to_path_buf(),
        stage: inputs
            .get("stage")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        method,
        backend,
        seed: inputs.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
        n_chains: inputs.get("n_chains").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        best_loglik: inputs.get("best_loglik").and_then(|v| v.as_f64()),
        best_chain: inputs
            .get("best_chain")
            .and_then(|v| v.as_u64())
            .map(|x| x as usize),
        axes: derive_axes(seg, dir),
    })
}

impl FitView {
    /// Project a CAS fit segment (`fits/{stem}-{h8}/`) into a fit-level view.
    /// Collects its `FitStage` leaves, sorts by stage label (`NN-` ordinal →
    /// execution order), reads the fit-level sidecar, and folds them.
    ///
    /// Provenance integrity (gh#147): a segment with stage leaves but no
    /// sidecar is malformed — skipped with a loud error rather than surfaced
    /// with empty provenance. For a Bayesian fit (`pgas`/`pmmh` leaf), an empty
    /// `resolved_priors` is a dropped-provenance bug signal (gh#75) and is
    /// flagged. Returns `None` when `seg` holds no `FitStage` leaves.
    pub fn read(seg: &Path) -> Option<FitView> {
        let mut leaves: Vec<(PathBuf, RunRecord)> = crate::cas_read::walk_records(seg)
            .into_iter()
            .filter(|(_, r)| r.kind == ArtifactKind::FitStage)
            .collect();
        if leaves.is_empty() {
            return None;
        }
        // Execution order: the `NN-stage` ordinal prefix sorts topologically.
        leaves.sort_by(|a, b| stage_label(&a.1).cmp(&stage_label(&b.1)));

        let fit_hash = leaves
            .iter()
            .find_map(|(_, r)| {
                r.levels
                    .iter()
                    .find(|l| l.name == "fit")
                    .map(|l| l.hash.to_hex())
            })
            .unwrap_or_default();
        let created_at = leaves
            .iter()
            .filter_map(|(_, r)| r.provenance.created_at.clone())
            .max()
            .unwrap_or_default();
        let engine_version = leaves
            .first()
            .map(|(_, r)| r.engine_version.clone())
            .unwrap_or_default();
        let argv = leaves
            .first()
            .map(|(_, r)| r.provenance.argv.clone())
            .unwrap_or_default();
        // Bare stage names in execution order, dedup preserving order.
        let mut stages_declared: Vec<String> = Vec::new();
        for (_, r) in &leaves {
            let bare = bare_stage_name(&stage_label(r));
            if !stages_declared.contains(&bare) {
                stages_declared.push(bare);
            }
        }

        // Provenance: from the sidecar, never defaulted. A fit with leaves but
        // no sidecar is malformed — skip it loudly (the writer always writes
        // one).
        let side = match crate::run_meta::read_fit_sidecar(seg) {
            Some(s) => s,
            None => {
                eprintln!(
                    "warning: fit {} has stage leaves but no fit.meta.json provenance \
                     sidecar — skipping (provenance missing)",
                    seg.display()
                );
                return None;
            }
        };
        // A Bayesian fit (`pgas`/`pmmh` leaf consumes priors) with empty
        // resolved_priors is a dropped-provenance bug (gh#75), not a valid
        // state.
        let is_bayesian = leaves.iter().any(|(_, r)| {
            matches!(
                r.inputs.get("method").and_then(|m| m.as_str()),
                Some("pgas") | Some("pmmh")
            )
        });
        if is_bayesian && side.resolved_priors.is_empty() {
            eprintln!(
                "error: Bayesian fit {} has empty resolved_priors in its sidecar — \
                 prior-source provenance was dropped (gh#75)",
                seg.display()
            );
        }

        let stages: Vec<FitStageView> = leaves
            .iter()
            .filter_map(|(dir, r)| stage_view_from_record(seg, dir, r))
            .collect();

        Some(FitView {
            fit_hash,
            engine_version,
            created_at,
            argv,
            label: side.label,
            model: side.model_path,
            model_identity: side.model_identity,
            fit_toml_path: side.fit_toml_path,
            fit_toml_hash: side.fit_toml_hash,
            data_hashes: side.data_hashes,
            estimated: side.estimated,
            fixed: side.fixed,
            resolved_priors: side.resolved_priors,
            parameters_provenance: side.parameters_provenance,
            stages_declared,
            stages,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_meta::{write_fit_sidecar, FitSidecar};

    /// Write a realistic two-stage CAS fit segment under `seg`:
    /// `{01-scout,02-refine}-<h8>/seed_1-<h8>/run.json` `FitStage` leaves plus
    /// the fit-level `fit.meta.json` sidecar. Mirrors the on-disk shape the
    /// runner writes. `fit_h8` seeds the shared `fit`-level hash.
    fn write_two_stage_fit(seg: &Path, fit_h8: &str) {
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let stages = [
            // (ordinal, stage label, bare stage, method, best_loglik, best_chain, created_at)
            (1_u8, "01-scout", "scout", "if2", -120.5_f64, 2_u64, "2026-04-27T00:00:01Z"),
            (2_u8, "02-refine", "refine", "if2", -56.7_f64, 1_u64, "2026-04-27T00:00:02Z"),
        ];
        for (ord, label, stage, method, best_ll, best_chain, created_at) in stages {
            let leaf = seg.join(format!("{label}-1fb03eee")).join("seed_1-06cbd6b3");
            std::fs::create_dir_all(&leaf).unwrap();
            // run_id must be valid 64-char hex (ContentHash rejects non-hex).
            let run_id = format!("{:0<64}", format!("{fit_h8}0{ord}"));
            let rec = format!(
                r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"demo","hash":"{fit_hash}","schema_version":1}},{{"name":"stage","label":"{label}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"stage":"{stage}","method":"{method}","backend":"chain_binomial","seed":1,"n_chains":4,"best_loglik":{best_ll},"best_chain":{best_chain}}},"provenance":{{"created_at":"{created_at}","argv":["camdl","fit","run"]}}}}"#
            );
            std::fs::write(leaf.join("run.json"), rec).unwrap();
        }
        let sidecar = FitSidecar {
            label: Some("a fit".into()),
            model_path: "sir.camdl".into(),
            model_identity: "f00d".repeat(16),
            fit_toml_path: "fit.toml".into(),
            fit_toml_hash: "cafe".repeat(16),
            data_hashes: HashMap::from([("cases".to_string(), "d4ta".repeat(2))]),
            estimated: vec!["beta".into(), "gamma".into()],
            fixed: HashMap::from([("N0".to_string(), 1000.0)]),
            resolved_priors: vec![],
            parameters_provenance: HashMap::new(),
            schema: None,
            docs: Default::default(),
        };
        // No fit.toml on disk → archive step is skipped; the sidecar still writes.
        write_fit_sidecar(seg, Path::new("nonexistent.toml"), &sidecar).unwrap();
    }

    /// Pins the N→1 aggregation `FitView` performs over a realistic two-stage
    /// fixture: the fit-level fold (fit_hash, latest created_at, engine/argv,
    /// sidecar provenance, execution-order `stages_declared`) and the per-stage
    /// fold (stage / method / backend / seed / n_chains / best_loglik /
    /// best_chain). Asserts real values — not `None == None` — so a degenerate
    /// projection (all-empty / all-None) fails.
    #[test]
    fn fit_view_folds_segment_field_for_field() {
        let tmp = crate::test_support::unique_temp_dir("fit_view_equiv");
        let seg = tmp.join("fits").join("demo-abcd1234");
        write_two_stage_fit(&seg, "abcd1234");

        let view = FitView::read(&seg).expect("FitView::read must derive a fit");

        // Fit-level identity / provenance.
        assert_eq!(view.fit_hash, "abcd1234".to_string() + &"0".repeat(56), "fit_hash");
        assert!(!view.fit_hash.is_empty(), "fit_hash must be non-empty");
        assert_eq!(view.created_at, "2026-04-27T00:00:02Z", "latest of the two leaves");
        assert_eq!(view.engine_version, "0.1.0+test", "engine_version (first leaf)");
        assert_eq!(view.argv, vec!["camdl", "fit", "run"], "argv (first leaf)");
        assert_eq!(view.label.as_deref(), Some("a fit"), "label (sidecar)");

        // Sidecar provenance.
        assert_eq!(view.model, "sir.camdl", "model");
        assert_eq!(view.model_identity, "f00d".repeat(16), "model_identity");
        assert!(!view.model_identity.is_empty(), "model_identity must be non-empty");
        assert_eq!(view.fit_toml_path, "fit.toml", "fit_toml_path");
        assert_eq!(view.fit_toml_hash, "cafe".repeat(16), "fit_toml_hash");
        assert_eq!(view.data_hashes.get("cases").map(String::as_str), Some("d4tad4ta"), "data_hashes");
        assert_eq!(view.estimated, vec!["beta", "gamma"], "estimated");
        assert_eq!(view.fixed.get("N0"), Some(&1000.0), "fixed");

        // stages_declared: execution order, deduped.
        assert_eq!(view.stages_declared, vec!["scout", "refine"], "execution order");

        // Per-stage fold, leaf-for-leaf.
        assert_eq!(view.stages.len(), 2, "two stage leaves");
        let scout = view.stages.iter().find(|s| s.stage == "scout").unwrap();
        assert_eq!(scout.method, FitAlgorithm::If2, "scout method");
        assert_eq!(scout.backend, InferenceBackend::ChainBinomial, "scout backend");
        assert_eq!(scout.seed, 1, "scout seed");
        assert_eq!(scout.n_chains, 4, "scout n_chains");
        assert_eq!(scout.best_loglik, Some(-120.5), "scout best_loglik");
        assert_eq!(scout.best_chain, Some(2), "scout best_chain");
        let refine = view.stages.iter().find(|s| s.stage == "refine").unwrap();
        assert_eq!(refine.best_loglik, Some(-56.7), "refine best_loglik");
        assert_eq!(refine.best_chain, Some(1), "refine best_chain");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
