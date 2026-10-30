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

use crate::chain_index::ChainIndex;
use crate::fit::fit_tree::{derive_axes, StageAxes};
use crate::run_meta::{InferenceBackend, FitAlgorithm, ParameterProvenance, ResolvedPriorEntry};

/// One fit-stage leaf, projected to the headline numbers the fit consumers
/// read directly from the leaf's `RunRecord` (`inputs` + path-derived `axes`).
#[derive(Debug, Clone)]
pub struct FitStageView {
    /// Stage directory holding this leaf's `run.json` (the consumer loads the
    /// typed θ̂ / diagnostics from sibling `fit_state.toml` etc. itself).
    pub stage_dir: PathBuf,
    /// Inference algorithm tag. Its `as_str()` is also the leaf's label —
    /// the `method` store level's directory name — so there is no separate
    /// name field to drift from it (gh#901).
    pub method: FitAlgorithm,
    /// The `method` level's full 64-char hash; its first 8 chars are the
    /// `<h8>` in `<method>-<h8>/` on disk.
    ///
    /// This is what tells two leaves of one algorithm apart (gh#895). Before
    /// the `[stages]` → `[method]` split a segment held one leaf per stage
    /// *name* and the name was the key; after it a segment holds one leaf per
    /// method *hash*, so two PGAS leaves differing only in `sweeps` have
    /// identical `method`, `backend` and `seed` and are distinguished by
    /// nothing else. `""` when the record carries no `method` level.
    pub method_hash: String,
    /// The leaf's own `run_id` (full 64-char hex) — the address `camdl show`
    /// and `fit predict` accept.
    pub run_id: String,
    /// The run's label (`run.json` `provenance.label`): the one run `@label`
    /// names (gh#896). `None` when the run was not given one.
    pub label: Option<String>,
    /// When the run was written (`provenance.created_at`, ISO 8601 UTC).
    pub created_at: String,
    /// Simulation backend the stage ran on. Defaults to `ChainBinomial` when
    /// absent.
    pub backend: InferenceBackend,
    pub seed: u64,
    pub n_chains: usize,
    pub best_loglik: Option<f64>,
    /// `inputs.best_chain` as recorded: the chain that won, naming a
    /// `chain_N/` directory under `stage_dir` (gh#912, gh#990). Renderers
    /// print it as it stands. `None` for a method with replicates rather than
    /// competing chains (pfilter), and on a leaf written before the field
    /// existed.
    pub best_chain: Option<ChainIndex>,
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
    /// The method labels found among the leaves, in label order,
    /// deduplicated. One per method that ran on this problem.
    pub methods_declared: Vec<String>,
    /// One view per discovered method leaf, sorted by method label then seed.
    pub stages: Vec<FitStageView>,
}

/// The `method` level's readable label (the algorithm's name); `""` when
/// absent.
fn stage_label(r: &RunRecord) -> String {
    r.levels
        .iter()
        .find(|l| l.name == "method")
        .map(|l| l.label.clone())
        .unwrap_or_default()
}

/// The `seed` level's label, for a stable order among one method's leaves.
fn seed_label(r: &RunRecord) -> String {
    r.levels
        .iter()
        .find(|l| l.name == "seed")
        .map(|l| l.label.clone())
        .unwrap_or_default()
}

/// The one stderr line for a fit leaf the view cannot read and skips, so a
/// stored fit that `fit summary` / `fit table` / `fit predict` leave out says
/// why instead of vanishing.
fn skipped_leaf_line(dir: &Path, why: &str) -> String {
    format!("warning: skipping fit leaf {}: {why}", dir.display())
}

/// Build a [`FitStageView`] from a fit-stage leaf record + its directory.
/// Errs, with the reason, when the record is not a `FitStage` or its `inputs`
/// lacks a `method` this camdl knows — a leaf written by a removed algorithm
/// (`nuts` on the ODE backend) is the case that matters. The caller says so
/// on stderr ([`skipped_leaf_line`]) rather than dropping the leaf silently.
fn stage_view_from_record(seg: &Path, dir: &Path, rec: &RunRecord) -> Result<FitStageView, String> {
    if rec.kind != ArtifactKind::FitStage {
        return Err("not a fit-stage record".into());
    }
    let inputs = rec
        .inputs
        .as_object()
        .ok_or_else(|| "its run.json has no `inputs` object".to_string())?;
    let method: FitAlgorithm = match inputs.get("method") {
        None => return Err("its run.json records no `method`".into()),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| {
            format!(
                "it records method {v}, which this camdl does not know (a method \
                 removed since the fit ran is read by the camdl that wrote it)"
            )
        })?,
    };
    let backend: InferenceBackend = inputs
        .get("backend")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or(InferenceBackend::ChainBinomial);
    Ok(FitStageView {
        stage_dir: dir.to_path_buf(),
        method,
        method_hash: rec
            .levels
            .iter()
            .find(|l| l.name == "method")
            .map(|l| l.hash.to_hex())
            .unwrap_or_default(),
        run_id: rec.run_id.to_hex(),
        label: rec.provenance.label.clone(),
        created_at: rec.provenance.created_at.clone().unwrap_or_default(),
        backend,
        seed: inputs.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
        n_chains: inputs.get("n_chains").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        best_loglik: inputs.get("best_loglik").and_then(|v| v.as_f64()),
        best_chain: inputs
            .get("best_chain")
            .and_then(|v| v.as_u64())
            .map(|x| ChainIndex(x as usize)),
        axes: derive_axes(seg, dir),
    })
}

/// The segment a fit leaf directory sits in (`<seg>/<algorithm>-<h8>/seed_<n>-<h8>`),
/// or `None` when `dir` is not a fit leaf (holds no `FitStage` `run.json`).
pub fn leaf_segment(dir: &Path) -> Option<PathBuf> {
    let bytes = std::fs::read(dir.join("run.json")).ok()?;
    let rec: RunRecord = serde_json::from_slice(&bytes).ok()?;
    if rec.kind != ArtifactKind::FitStage {
        return None;
    }
    let lexical = dir.parent().and_then(Path::parent).filter(|p| !p.as_os_str().is_empty());
    match lexical {
        Some(seg) => Some(seg.to_path_buf()),
        None => std::fs::canonicalize(dir).ok()?.parent()?.parent().map(Path::to_path_buf),
    }
}

/// Where `fit predict` writes for one run: `<segment>/predict/<run id, 12 hex>/`
/// (gh#896), or `<run id, 12 hex>-<options key>/` when the predict was run
/// with non-default options (`fit::predict::PredictOptions::address_key`).
/// Keyed by the run, so two runs of one problem keep separate predictives,
/// and by the options, so a `--n-draws 3` predict does not overwrite the
/// default one; outside the leaf, whose exact-set manifest admits no
/// regenerated files. The full run id and the options are recorded in the
/// manifests written there (`predictive.json`, `observed.json`, `report.json`).
pub fn predictions_dir(segment: &Path, run_id: &str, options_key: Option<&str>) -> PathBuf {
    let run = &run_id[..12.min(run_id.len())];
    segment.join("predict").join(match options_key {
        Some(k) => format!("{run}-{k}"),
        None => run.to_string(),
    })
}

/// The `run_id` (full hex) a fit leaf's `run.json` records, when `dir` is one.
pub fn leaf_run_id(dir: &Path) -> Option<String> {
    let bytes = std::fs::read(dir.join("run.json")).ok()?;
    let rec: RunRecord = serde_json::from_slice(&bytes).ok()?;
    (rec.kind == ArtifactKind::FitStage).then(|| rec.run_id.to_hex())
}

/// Whether two paths name one directory, lexically or after resolving links.
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b
        || match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
            (Ok(x), Ok(y)) => x == y,
            _ => false,
        }
}

impl FitView {
    /// The labels of this problem's runs, in run order, joined for display —
    /// `None` when no run is labelled. A label names one run, not the problem
    /// (gh#896), so a folder with several labelled runs shows each.
    pub fn label(&self) -> Option<String> {
        let labels: Vec<&str> = self.stages.iter().filter_map(|s| s.label.as_deref()).collect();
        (!labels.is_empty()).then(|| labels.join(", "))
    }

    /// Project a CAS fit segment (`fits/{stem}-{h8}/`) into a fit-level view.
    /// Collects its `FitStage` leaves, sorts by method label then seed, reads
    /// the fit-level sidecar, and folds them.
    ///
    /// Provenance integrity (gh#147): a segment with stage leaves but no
    /// sidecar is malformed — skipped with a loud error rather than surfaced
    /// with empty provenance. For a Bayesian fit (`pgas`/`pmmh` leaf), an empty
    /// `resolved_priors` is a dropped-provenance bug signal (gh#75) and is
    /// flagged. Returns `None` when `seg` holds no `FitStage` leaves.
    ///
    /// A leaf directory reads as the view of that one run (gh#896): the
    /// problem's provenance from the segment the leaf sits in, and `stages`
    /// holding the leaf alone. Every reader that takes a fit directory
    /// therefore reads exactly the run a leaf path names, never a sibling.
    pub fn read(dir: &Path) -> Option<FitView> {
        let (seg, only): (PathBuf, Option<&Path>) = match leaf_segment(dir) {
            Some(seg) => (seg, Some(dir)),
            None => (dir.to_path_buf(), None),
        };
        let seg = seg.as_path();
        let mut leaves: Vec<(PathBuf, RunRecord)> = crate::cas_read::walk_records(seg)
            .into_iter()
            .filter(|(_, r)| r.kind == ArtifactKind::FitStage)
            .filter(|(d, _)| only.is_none_or(|o| same_dir(d, o)))
            .collect();
        if leaves.is_empty() {
            return None;
        }
        // Label order, then seed: there is no execution order among methods.
        leaves.sort_by(|a, b| {
            (stage_label(&a.1), seed_label(&a.1)).cmp(&(stage_label(&b.1), seed_label(&b.1)))
        });

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
        // Method labels in label order, dedup preserving order.
        let mut methods_declared: Vec<String> = Vec::new();
        for (_, r) in &leaves {
            let label = stage_label(r);
            if !methods_declared.contains(&label) {
                methods_declared.push(label);
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
            .filter_map(|(dir, r)| match stage_view_from_record(seg, dir, r) {
                Ok(v) => Some(v),
                Err(why) => {
                    eprintln!("{}", skipped_leaf_line(dir, &why));
                    None
                }
            })
            .collect();

        Some(FitView {
            fit_hash,
            engine_version,
            created_at,
            argv,
            model: side.model_path,
            model_identity: side.model_identity,
            fit_toml_path: side.fit_toml_path,
            fit_toml_hash: side.fit_toml_hash,
            data_hashes: side.data_hashes.into_iter().collect(),
            estimated: side.estimated,
            fixed: side.fixed.into_iter().collect(),
            resolved_priors: side.resolved_priors,
            parameters_provenance: side.parameters_provenance.into_iter().collect(),
            methods_declared,
            stages,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leaf whose run.json names an algorithm this camdl does not know (a
    /// stored NUTS-on-ODE fit, after that method's removal) is skipped with a
    /// line naming the leaf and the reason, not dropped silently.
    #[test]
    fn a_leaf_with_an_unknown_algorithm_is_skipped_with_a_reason() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_unknown_algo_{}_{}", std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let leaf = dir.join("nuts-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let rec: RunRecord = serde_json::from_str(&format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"method","label":"nuts","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"nuts","backend":"ode","seed":1,"n_chains":2}},"provenance":{{"created_at":"2026-07-07T00:00:00Z","argv":["camdl","fit","run"]}}}}"#,
            run_id = "ab".repeat(32),
        ))
        .unwrap();
        let why = stage_view_from_record(&dir, &leaf, &rec)
            .expect_err("`nuts` is not a FitAlgorithm any more");
        let line = skipped_leaf_line(&leaf, &why);
        assert!(line.starts_with("warning: skipping fit leaf "), "{line}");
        assert!(line.contains(&leaf.display().to_string()), "names the leaf: {line}");
        assert!(line.contains("\"nuts\"") && line.contains("does not know"),
            "names the method and the reason: {line}");
        std::fs::remove_dir_all(&dir).ok();
    }
    use crate::run_meta::{write_fit_sidecar, FitSidecar};

    /// Write a realistic two-method CAS fit segment under `seg` — the shape
    /// two files with one problem half produce:
    /// `{if2,pgas}-<h8>/seed_1-<h8>/run.json` `FitStage` leaves plus the
    /// fit-level `fit.meta.json` sidecar. Mirrors the on-disk shape the
    /// runner writes. `fit_h8` seeds the shared `fit`-level hash.
    fn write_two_stage_fit(seg: &Path, fit_h8: &str) {
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let stages = [
            // (ordinal, method label, method, best_loglik, best_chain, created_at)
            (1_u8, "if2", "if2", -120.5_f64, 2_u64, "2026-04-27T00:00:01Z"),
            (2_u8, "pgas", "pgas", -56.7_f64, 1_u64, "2026-04-27T00:00:02Z"),
        ];
        for (ord, label, method, best_ll, best_chain, created_at) in stages {
            let leaf = seg.join(format!("{label}-1fb03eee")).join("seed_1-06cbd6b3");
            std::fs::create_dir_all(&leaf).unwrap();
            // run_id must be valid 64-char hex (ContentHash rejects non-hex).
            let run_id = format!("{:0<64}", format!("{fit_h8}0{ord}"));
            let rec = format!(
                r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"demo","hash":"{fit_hash}","schema_version":1}},{{"name":"method","label":"{label}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"{method}","backend":"chain_binomial","seed":1,"n_chains":4,"best_loglik":{best_ll},"best_chain":{best_chain}}},"provenance":{{"created_at":"{created_at}","argv":["camdl","fit","run"]}}}}"#
            );
            std::fs::write(leaf.join("run.json"), rec).unwrap();
        }
        let sidecar = FitSidecar {
            label: Some("a fit".into()),
            model_path: "sir.camdl".into(),
            model_identity: "f00d".repeat(16),
            fit_toml_path: "fit.toml".into(),
            fit_toml_hash: "cafe".repeat(16),
            data_hashes: std::collections::BTreeMap::from([("cases".to_string(), "d4ta".repeat(2))]),
            estimated: vec!["beta".into(), "gamma".into()],
            fixed: std::collections::BTreeMap::from([("N0".to_string(), 1000.0)]),
            partition: None,
            resolved_priors: vec![crate::run_meta::ResolvedPriorEntry {
                param: "beta".into(),
                source: "model_ir".into(),
                overridden_model_prior: None,
            }],
            parameters_provenance: std::collections::BTreeMap::new(),
            schema: None,
            docs: Default::default(),
            training_window: None,
        };
        // No fit.toml on disk → archive step is skipped; the sidecar still writes.
        write_fit_sidecar(seg, Path::new("nonexistent.toml"), &sidecar).unwrap();
    }

    /// Pins the N→1 aggregation `FitView` performs over a realistic
    /// two-method fixture: the fit-level fold (fit_hash, latest created_at,
    /// engine/argv, sidecar provenance, label-order `methods_declared`) and the
    /// per-leaf fold (label / method / backend / seed / n_chains /
    /// best_loglik / best_chain). Asserts real values — not `None == None` —
    /// so a degenerate projection (all-empty / all-None) fails.
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
        assert_eq!(view.label(), None, "no run is labelled; the sidecar's label is not a run's");

        // Sidecar provenance.
        assert_eq!(view.model, "sir.camdl", "model");
        assert_eq!(view.model_identity, "f00d".repeat(16), "model_identity");
        assert!(!view.model_identity.is_empty(), "model_identity must be non-empty");
        assert_eq!(view.fit_toml_path, "fit.toml", "fit_toml_path");
        assert_eq!(view.fit_toml_hash, "cafe".repeat(16), "fit_toml_hash");
        assert_eq!(view.data_hashes.get("cases").map(String::as_str), Some("d4tad4ta"), "data_hashes");
        assert_eq!(view.estimated, vec!["beta", "gamma"], "estimated");
        assert_eq!(view.fixed.get("N0"), Some(&1000.0), "fixed");

        // methods_declared: label order, deduped.
        assert_eq!(view.methods_declared, vec!["if2", "pgas"], "label order");

        // Per-leaf fold, leaf-for-leaf.
        assert_eq!(view.stages.len(), 2, "two method leaves");
        let scout = view.stages.iter().find(|s| s.method == FitAlgorithm::If2).unwrap();
        assert_eq!(scout.method, FitAlgorithm::If2, "if2 method");
        assert_eq!(scout.backend, InferenceBackend::ChainBinomial, "if2 backend");
        assert_eq!(scout.seed, 1, "if2 seed");
        assert_eq!(scout.n_chains, 4, "if2 n_chains");
        assert_eq!(scout.best_loglik, Some(-120.5), "if2 best_loglik");
        assert_eq!(scout.best_chain, Some(ChainIndex(2)), "if2 best_chain");
        let refine = view.stages.iter().find(|s| s.method == FitAlgorithm::Pgas).unwrap();
        assert_eq!(refine.method, FitAlgorithm::Pgas, "pgas method");
        assert_eq!(refine.best_loglik, Some(-56.7), "pgas best_loglik");
        assert_eq!(refine.best_chain, Some(ChainIndex(1)), "pgas best_chain");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// gh#896: a leaf directory is a fit view of that one run. The problem's
    /// provenance comes from the segment the leaf sits in; the runs are the
    /// leaf alone, so a reader handed a leaf reads that run and no sibling.
    #[test]
    fn a_leaf_directory_reads_as_the_view_of_that_one_run() {
        let tmp = crate::test_support::unique_temp_dir("fit_view_leaf");
        let seg = tmp.join("fits").join("demo-abcd1234");
        write_two_stage_fit(&seg, "abcd1234");
        let leaf = seg.join("if2-1fb03eee").join("seed_1-06cbd6b3");

        let view = FitView::read(&leaf).expect("a leaf directory is a fit view");
        assert_eq!(view.stages.len(), 1, "the leaf alone, not its sibling");
        assert_eq!(view.stages[0].method, FitAlgorithm::If2);
        assert_eq!(view.stages[0].stage_dir, leaf);
        assert_eq!(view.methods_declared, vec!["if2"]);
        assert_eq!(view.created_at, "2026-04-27T00:00:01Z", "this run's time, not the latest");
        // Problem provenance is the segment's.
        assert_eq!(view.model, "sir.camdl");
        assert_eq!(view.estimated, vec!["beta", "gamma"]);
        assert_eq!(view.fit_hash, "abcd1234".to_string() + &"0".repeat(56));

        std::fs::remove_dir_all(&tmp).ok();
    }
}
