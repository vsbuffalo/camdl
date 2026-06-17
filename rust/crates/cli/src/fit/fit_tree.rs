//! Canonical fit-directory walker.
//!
//! One module, two functions, used by every consumer that needs to
//! enumerate fit-stage runs:
//!
//! - [`walk_fit_dir`] — given one fit_dir (`results/fits/<stem>-<hash>/`)
//!   returns one [`StageNode`] per `FitStage` `runid::RunRecord` leaf found
//!   underneath, each carrying the leaf's [`FitStageView`] (method, seed,
//!   best_loglik, …) and its path-derived axes.
//! - [`walk_fits_root`] — given the top-level `results/fits/` returns one
//!   [`FitDirEntry`] per fit segment, each carrying its already-parsed
//!   [`FitView`] (fit-level identity + provenance + stage views) so callers
//!   filter by model_hash / label / declared stages without a second read.
//!
//! `StageNode`/`FitStageView` are method-agnostic by construction: they carry
//! no `fit_state_path` (an IF2-only artifact). Consumers that need the typed
//! result load it via `MethodResult::load_from`.

use std::io;
use std::path::{Path, PathBuf};

use crate::fit::fit_view::{FitStageView, FitView};

/// One discovered fit-stage run inside a fit_dir.
///
/// `axes` (on the stage view) carries the (data_kind, fit_seed, sweep_slug)
/// triple extracted from the path relative to the enclosing fit_dir. When the
/// path doesn't fit the canonical v2 layout, `axes` is `None` — the stage is
/// still surfaced (its leaf is real) but consumers that need axis grouping skip
/// it.
#[derive(Debug, Clone)]
pub struct StageNode {
    /// Absolute (or whatever the caller passed) path to the stage directory
    /// containing the leaf `run.json`.
    pub stage_dir: PathBuf,
    /// The stage's projected new-format view.
    pub stage: FitStageView,
}

/// (data_kind, fit_seed, sweep_slug) triple extracted from a stage's
/// path relative to its fit_dir.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAxes {
    pub data_kind: DataKind,
    pub fit_seed: u64,
    /// `None` when no `--sweep` was active for this fit. When the sweep
    /// produced a single point with no slug, also `None`.
    pub sweep_slug: Option<String>,
}

/// Whether this stage fit a real-data dataset or one of N synthetic
/// replicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataKind {
    Real,
    Synthetic { ds_idx: usize },
}

/// One fit segment returned by [`walk_fits_root`]. The `view` field is
/// pre-parsed so `fit table`'s outer loop can filter by model_hash / label /
/// declared stages without re-reading the leaves per row.
#[derive(Debug, Clone)]
pub struct FitDirEntry {
    pub fit_dir: PathBuf,
    /// The fit-level view (identity + provenance + per-stage views).
    pub view: FitView,
}

/// Walk one fit directory and return every `FitStage` leaf found underneath,
/// each as a [`StageNode`]. Filesystem layout is intentionally not constrained
/// beyond "every stage writes a `runid::RunRecord` leaf": this walker works on
/// real-data fits, synthetic-replicate fits, sweep cells, any combination, and
/// on user-defined non-canonical layouts.
///
/// Returns runs in lexicographic order on `stage_dir` (deterministic for
/// tests). A segment with stage leaves but no provenance sidecar is skipped
/// loudly (see [`FitView::read`]); the function then yields no nodes.
pub fn walk_fit_dir(fit_dir: &Path) -> io::Result<Vec<StageNode>> {
    if !fit_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("fit_dir not found: {}", fit_dir.display()),
        ));
    }
    let mut out: Vec<StageNode> = match FitView::read(fit_dir) {
        Some(view) => view
            .stages
            .into_iter()
            .map(|stage| StageNode {
                stage_dir: stage.stage_dir.clone(),
                stage,
            })
            .collect(),
        None => Vec::new(),
    };
    out.sort_by(|a, b| a.stage_dir.cmp(&b.stage_dir));
    Ok(out)
}

/// Walk `results/fits/` (or whatever root the caller passes) and return one
/// entry per fit segment. A child dir is a fit segment iff it holds `FitStage`
/// leaves and a provenance sidecar ([`FitView::read`]); other dirs are skipped.
///
/// Returns entries in lexicographic order on `fit_dir`.
pub fn walk_fits_root(root: &Path) -> io::Result<Vec<FitDirEntry>> {
    let mut out = Vec::new();
    if !root.exists() {
        // Empty root → empty list. `walk_fit_dir` errors on a missing dir
        // because that's a single-fit lookup; this is a directory listing, and
        // "no fits yet" is a normal state.
        return Ok(out);
    }
    // gh#147: a CAS fit is a segment dir `fits/{stem}-{h8}/` holding its
    // stage-leaf subdirs (`{NN-stage}-{h8}/seed_N-{h8}/run.json`, kind
    // FitStage) plus the fit-level sidecar. `FitView::read` derives ONE
    // fit-level view per segment from those leaves — the same projection
    // `table_row::build_row` consumes, so the table row and its source entry
    // agree. A dir with no FitStage leaves is not a fit segment and is skipped.
    let entries = std::fs::read_dir(root)?;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(view) = FitView::read(&p) {
            out.push(FitDirEntry {
                fit_dir: p.clone(),
                view,
            });
        }
    }
    out.sort_by(|a, b| a.fit_dir.cmp(&b.fit_dir));
    Ok(out)
}

// ── internals ───────────────────────────────────────────────────────

/// Extract `StageAxes` from a stage path relative to its fit_dir.
/// Canonical v2 layouts:
///
/// ```text
/// <fit_dir>/real/fit_<seed>/<stage>/
/// <fit_dir>/real/fit_<seed>/<sweep_slug>/<stage>/
/// <fit_dir>/synthetic/ds_<NN>/fit_<seed>/<stage>/
/// <fit_dir>/synthetic/ds_<NN>/fit_<seed>/<sweep_slug>/<stage>/
/// ```
///
/// Returns `None` when the path doesn't fit either shape (e.g. a non-canonical
/// user layout or a profile sub-tree). The walker still surfaces those stages —
/// `axes` is None on the view and consumers that need axis grouping skip them.
pub(crate) fn derive_axes(fit_dir: &Path, stage_dir: &Path) -> Option<StageAxes> {
    let rel = stage_dir.strip_prefix(fit_dir).ok()?;
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // The last component is the stage name; we don't need it for axes
    // (it's already on the stage view's `stage`). What precedes it must match
    // one of the four shapes above.
    if parts.len() < 3 {
        return None;
    }
    let body = &parts[..parts.len() - 1];

    // Real layouts: ["real", "fit_<seed>"] or ["real", "fit_<seed>", "<slug>"]
    if body.first() == Some(&"real") {
        let fit_seed = parse_fit_seed(body.get(1)?)?;
        let sweep_slug = match body.len() {
            2 => None,
            3 => Some(body[2].to_string()),
            _ => return None,
        };
        return Some(StageAxes {
            data_kind: DataKind::Real,
            fit_seed,
            sweep_slug,
        });
    }

    // Synthetic layouts:
    //   ["synthetic", "ds_<NN>", "fit_<seed>"]
    //   ["synthetic", "ds_<NN>", "fit_<seed>", "<slug>"]
    if body.first() == Some(&"synthetic") {
        let ds_idx = parse_ds_idx(body.get(1)?)?;
        let fit_seed = parse_fit_seed(body.get(2)?)?;
        let sweep_slug = match body.len() {
            3 => None,
            4 => Some(body[3].to_string()),
            _ => return None,
        };
        return Some(StageAxes {
            data_kind: DataKind::Synthetic { ds_idx },
            fit_seed,
            sweep_slug,
        });
    }

    None
}

fn parse_fit_seed(s: &str) -> Option<u64> {
    s.strip_prefix("fit_")?.parse().ok()
}

fn parse_ds_idx(s: &str) -> Option<usize> {
    s.strip_prefix("ds_")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_meta::FitAlgorithm;

    /// Allocate a unique tempdir for one test. Cleaned up by `Drop`.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_fittree_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    /// Write a fit-level sidecar (`fit.meta.json`) under a segment dir so
    /// `FitView::read` treats it as a well-formed fit. Minimal provenance —
    /// `stages_declared`/per-stage numbers come from the leaves.
    fn write_sidecar(seg: &Path) {
        std::fs::create_dir_all(seg).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_path":"sir.camdl","model_hash":"f00d","fit_toml_path":"fit.toml"}"#,
        )
        .unwrap();
    }

    /// Place a `FitStage` `runid::RunRecord` leaf at `<fit_dir>/<rel>/run.json`.
    /// `ord` seeds a valid-hex run_id; `stage`/`method`/`seed` go in `inputs`.
    fn place_stage(fit_dir: &Path, rel: &str, ord: u32, stage: &str, method: &str, seed: u64) {
        let dir = fit_dir.join(rel);
        std::fs::create_dir_all(&dir).unwrap();
        let run_id = format!("{:0<64}", format!("ae{ord:06x}"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"deadbeef00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"stage","label":"{stage}","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_{seed}","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"stage":"{stage}","method":"{method}","backend":"chain_binomial","seed":{seed},"n_chains":2,"best_loglik":-100.0,"best_chain":0}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(dir.join("run.json"), rec).unwrap();
    }

    /// gh#147: write a CAS fit segment at `seg` — one FitStage leaf
    /// (`01-mle-<h8>/seed_1-<h8>/run.json`) plus the fit-level sidecar — the
    /// shape `walk_fits_root` reads. `fit_h8` (8 hex) seeds the shared
    /// `fit`-level hash so each segment gets a distinct fit hash.
    fn write_cas_fit_seg(seg: &Path, fit_h8: &str) {
        let leaf = seg.join("01-mle-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let run_id = format!("{:0<64}", format!("{fit_h8}01"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"{fit_hash}","schema_version":1}},{{"name":"stage","label":"01-mle","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"stage":"mle","method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(leaf.join("run.json"), rec).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_hash":"f00d","model_path":"sir.camdl","fit_toml_path":"fit.toml"}"#,
        )
        .unwrap();
    }

    #[test]
    fn walks_real_only_single_fit() {
        let tmp = tempdir("real_only");
        let fit_dir = tmp.path().join("fit_he-deadbeef");
        write_sidecar(&fit_dir);
        place_stage(&fit_dir, "real/fit_1/scout", 1, "scout", "if2", 1);
        place_stage(&fit_dir, "real/fit_1/refine", 2, "refine", "if2", 1);

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        // Both nodes carry the canonical real/fit_1 axes, no sweep.
        for node in &nodes {
            let axes = node.stage.axes.as_ref().expect("real fit must produce axes");
            assert_eq!(axes.data_kind, DataKind::Real);
            assert_eq!(axes.fit_seed, 1);
            assert_eq!(axes.sweep_slug, None);
        }
        let stage_names: Vec<&str> = nodes.iter().map(|n| n.stage.stage.as_str()).collect();
        assert!(stage_names.contains(&"scout"));
        assert!(stage_names.contains(&"refine"));
    }

    #[test]
    fn walks_synthetic_layout() {
        let tmp = tempdir("synthetic");
        let fit_dir = tmp.path().join("fit_syn-cafebabe");
        write_sidecar(&fit_dir);
        let mut ord = 0;
        for ds in 1..=2usize {
            for fs in [11u64, 22] {
                ord += 1;
                let rel = format!("synthetic/ds_{:02}/fit_{}/mle", ds, fs);
                place_stage(&fit_dir, &rel, ord, "mle", "if2", fs);
            }
        }
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 4, "2 datasets × 2 seeds = 4 nodes");
        // Spot-check one cell's axes.
        let ds01_22 = nodes
            .iter()
            .find(|n| {
                let axes = n.stage.axes.as_ref().unwrap();
                matches!(axes.data_kind, DataKind::Synthetic { ds_idx: 1 }) && axes.fit_seed == 22
            })
            .expect("ds_01 × fit_22 cell");
        assert_eq!(ds01_22.stage.axes.as_ref().unwrap().sweep_slug, None);
    }

    #[test]
    fn walks_sweep_slug_layout() {
        let tmp = tempdir("sweep");
        let fit_dir = tmp.path().join("fit_sweep-12345678");
        write_sidecar(&fit_dir);
        place_stage(&fit_dir, "real/fit_1/R0_1.000/mle", 1, "mle", "if2", 1);
        place_stage(&fit_dir, "real/fit_1/R0_2.000/mle", 2, "mle", "if2", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let slugs: Vec<String> = nodes
            .iter()
            .map(|n| n.stage.axes.as_ref().unwrap().sweep_slug.clone().unwrap())
            .collect();
        assert!(slugs.iter().any(|s| s == "R0_1.000"));
        assert!(slugs.iter().any(|s| s == "R0_2.000"));
    }

    #[test]
    fn walks_mixed_method_stages() {
        let tmp = tempdir("mixed");
        let fit_dir = tmp.path().join("fit_mixed-aaaabbbb");
        write_sidecar(&fit_dir);
        place_stage(&fit_dir, "real/fit_1/scout", 1, "scout", "if2", 1);
        place_stage(&fit_dir, "real/fit_1/pgas", 2, "pgas", "pgas", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let methods: Vec<&str> = nodes.iter().map(|n| n.stage.method.as_str()).collect();
        assert!(methods.contains(&"if2"));
        assert!(methods.contains(&"pgas"));
    }

    #[test]
    fn skips_malformed_run_json_does_not_panic() {
        let tmp = tempdir("malformed");
        let fit_dir = tmp.path().join("fit_bad-deadbeef");
        write_sidecar(&fit_dir);
        let stage_dir = fit_dir.join("real/fit_1/scout");
        std::fs::create_dir_all(&stage_dir).unwrap();
        // Stage run.json is garbage — `walk_records` drops it.
        std::fs::write(stage_dir.join("run.json"), "{ not valid json }").unwrap();

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert!(
            nodes.is_empty(),
            "malformed stage run.json should be skipped, got {:?}",
            nodes
        );
    }

    #[test]
    fn missing_fit_dir_errors() {
        let err = walk_fit_dir(Path::new("/definitely/does/not/exist/here")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn surfaces_non_canonical_layout_with_axes_none() {
        // A user might park a fit-stage leaf under a one-off path (e.g. for a
        // debugging copy). The walker should still surface it — `axes` is None
        // signals "not in canonical layout."
        let tmp = tempdir("noncanon");
        let fit_dir = tmp.path().join("fit_x-aabbccdd");
        write_sidecar(&fit_dir);
        place_stage(&fit_dir, "weird_layout/scout", 1, "scout", "if2", 1);
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(
            nodes[0].stage.axes.is_none(),
            "non-canonical layout must produce axes=None"
        );
    }

    #[test]
    fn walks_fits_root_returns_one_entry_per_fit() {
        let tmp = tempdir("root");
        let fits_root = tmp.path().join("fits");
        std::fs::create_dir_all(&fits_root).unwrap();

        let dirs = ["fit_a-11111111", "fit_b-22222222"];
        for name in dirs.iter() {
            let d = fits_root.join(name);
            // gh#147: a CAS fit segment (one FitStage leaf + the fit-level
            // sidecar) — the shape `walk_fits_root`/`FitView::read` reads.
            let h8 = name.split('-').next_back().unwrap();
            write_cas_fit_seg(&d, h8);
        }
        // A dir with no FitStage leaves (and no sidecar) — should be skipped.
        let intruder = fits_root.join("not_a_fit");
        std::fs::create_dir_all(&intruder).unwrap();
        std::fs::write(intruder.join("README"), b"not a fit").unwrap();

        let entries = walk_fits_root(&fits_root).unwrap();
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e.fit_dir.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["fit_a-11111111", "fit_b-22222222"]);
        // The view is pre-parsed (derived from the stage leaves); `stages_declared`
        // comes from the leaves' stage levels. gh#147: `estimated`/`fixed`/priors
        // are config detail not carried on the leaves, so they default empty —
        // config-diff reads the archived `fit.toml.original` for those.
        assert_eq!(entries[0].view.stages_declared, vec!["mle"]);
        assert!(!entries[0].view.fit_hash.is_empty());
        assert_eq!(entries[0].view.stages.len(), 1);
    }

    #[test]
    fn walks_fits_root_empty_when_no_root() {
        let entries = walk_fits_root(Path::new("/definitely/not/here/either/fits")).unwrap();
        assert!(entries.is_empty());
    }

    // Keep `FitAlgorithm` exercised so the import isn't unused in the
    // method-agnostic test set above.
    #[test]
    fn method_kind_str_roundtrip() {
        assert_eq!(FitAlgorithm::If2.as_str(), "if2");
        assert_eq!(FitAlgorithm::Pgas.as_str(), "pgas");
    }
}
